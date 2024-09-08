// std
use std::sync::Arc;
use std::collections::HashMap;

// WebRtc
use webrtc::api::interceptor_registry::register_default_interceptors;
use webrtc::api::media_engine::MediaEngine;
use webrtc::api::APIBuilder;
use webrtc::api::API;
use webrtc::data_channel::data_channel_message::DataChannelMessage;
use webrtc::data_channel::RTCDataChannel;
use webrtc::ice_transport::ice_candidate::{RTCIceCandidate, RTCIceCandidateInit};
use webrtc::ice_transport::ice_server::RTCIceServer;
use webrtc::interceptor::registry::Registry;
use webrtc::peer_connection::configuration::RTCConfiguration;
use webrtc::peer_connection::peer_connection_state::RTCPeerConnectionState;
use webrtc::peer_connection::sdp::session_description::RTCSessionDescription;
use webrtc::peer_connection::RTCPeerConnection;
use webrtc::ice_transport::ice_gathering_state::RTCIceGatheringState;

// Other
use anyhow::{anyhow, Result};
use parking_lot::Mutex;
use rand::Rng;
use crossbeam::channel::Receiver;
use crossbeam::channel::unbounded;
use async_trait::async_trait;
use async_std::task::block_on;
use bytes::Bytes;

// Local crates
use crate::WebRtcConnection;
use crate::WebRtcDataChannel;
use crate::WebRtcDataChannelType;
use crate::IceCandidate;
use crate::CallerType;
use crate::OfferAnswer;
use crate::ReconnectionState;
use crate::STUN_SERVERS;


// region: Types

pub type WebRtcGuardian = WebRtcConnectionNative;

// endregion

// region: Structs

pub struct WebRtcConnectionNative {
    id: u128,
    remote_id: Arc<Mutex<Option<u128>>>,
    caller_type: Arc<Mutex<CallerType>>,
    peer_connection: Arc<RTCPeerConnection>,
    local_ice_candidates: Arc<Mutex<Vec<IceCandidate>>>,
    data_channels: Arc<Mutex<HashMap<String, WebRtcDataChannelNative>>>,
    reconnection_state: Arc<Mutex<ReconnectionState>>,
}

pub struct WebRtcDataChannelNative {
    data_channel: Arc<RTCDataChannel>,
    msg_received: Receiver<Vec<u8>>
}

// endregion

// region: Implementations

impl WebRtcDataChannel for WebRtcDataChannelNative {

    fn new(data_channel: Arc<WebRtcDataChannelType>) -> Self {

        // Channels
        let (msg_incoming, msg_received) = unbounded::<Vec<u8>>();
        let msg_incoming_clone = msg_incoming.clone();

        // What to do when receiving a message from data_channel
        block_on(
            data_channel
            .on_message(
                Box::new(move |msg: DataChannelMessage| {
                    let msg = msg.data.to_vec();
                    msg_incoming_clone.try_send(msg).unwrap();
                    Box::pin(async { })
                })
            )
        );

        Self {
            data_channel,
            msg_received
        }
    }

    fn send(&self, data: &[u8]) -> Result<()> {
        let msg = Bytes::copy_from_slice(data);
        let sent = block_on(self.data_channel.send(&msg));
        match sent {
            Ok(_) => Ok(()),
            Err(e) =>  Err(anyhow!("WebRTC error: {e:?}"))
        }
    }

    fn receive(&self) -> Vec<Vec<u8>> {
        let mut messages = vec![];
        loop {
            match self.msg_received.try_recv() {
                Ok(msg) => {messages.push(msg);},
                Err(_) => break
            }
        }
        messages
    }

}

#[async_trait(?Send)]
impl WebRtcConnection for WebRtcConnectionNative {

    fn new() -> Result<Box<Self>> {

        // Create ID
        let mut rng = rand::thread_rng();
        let id: u128 = rng.gen();
        let remote_id = Arc::new(Mutex::new(None));
        let caller_type = Arc::new(Mutex::new(CallerType::Receiver));  // Always default to "receiver"
        let reconnection_state = Arc::new(Mutex::new(ReconnectionState { reconnection_initiated: false, reconnection_failed: false}));

        // Create config (api)
        let api = create_api().unwrap();
        let stun_servers: Vec<String> = STUN_SERVERS.iter().map(|s| s.to_string()).collect();
        let config = RTCConfiguration {
            ice_servers: vec![RTCIceServer {
                urls: stun_servers,
                ..Default::default()
            }],
            ..Default::default()
        };

        let peer_connection = Arc::new(
            block_on(api.new_peer_connection(config)).unwrap()  // Needs to block to be same as wasm
        );

        // Set method for gathering ICE candidates
        let local_ice_candidates = Arc::new(Mutex::new(Vec::<IceCandidate>::new()));
        let local_ice_candidates_clone = local_ice_candidates.clone();
        block_on(
            peer_connection
            .on_ice_candidate(Box::new(move |candidate: Option<RTCIceCandidate>| {
                let local_ice_candidates_clone2 = local_ice_candidates_clone.clone();
                Box::pin(async move {
                    if let Some(candidate) = candidate {
                        if let Ok(candidate) = candidate.to_json().await {
                            let ice_candidate = IceCandidate {
                                candidate: candidate.candidate,
                                sdp_mid: candidate.sdp_mid.unwrap_or("0".to_string()),
                                sdp_m_line_index: candidate.sdp_mline_index.unwrap_or(0)
                            };
                            local_ice_candidates_clone2.lock().push(ice_candidate);
                        }
                    }
                })
            }))
        );

        // A method to save received data channels and messages
        let data_channels = Arc::new(Mutex::new(HashMap::<String, WebRtcDataChannelNative>::new()));
        let data_channels_clone = data_channels.clone();
        block_on(
            peer_connection
            .on_data_channel(
                Box::new(move |data_channel: Arc<RTCDataChannel>| {
                    let data_channels_clone2 = data_channels_clone.clone();
                    Box::pin(async move {
                        let data_channel_label = data_channel.label().to_owned();
                        let webrtc_data_channel = WebRtcDataChannelNative::new(data_channel);
                        data_channels_clone2.lock().insert(data_channel_label, webrtc_data_channel);
                    })
                })
            )
        );

        // Needed?
        block_on(
            peer_connection
            .on_peer_connection_state_change(Box::new(move |s: RTCPeerConnectionState| {
                if s == RTCPeerConnectionState::Failed {
                    // Wait until PeerConnection has had no network activity for 30 seconds or another failure. It may be reconnected using an ICE Restart.
                    // Use webrtc.PeerConnectionStateDisconnected if you are interested in detecting faster timeout.
                    // Note that the PeerConnection may come back from PeerConnectionStateDisconnected.
                }
                Box::pin(async {})
            }))
        );

        Ok(
            Box::new(
                WebRtcConnectionNative {
                    id,
                    remote_id,
                    caller_type,
                    peer_connection,
                    local_ice_candidates,
                    data_channels,
                    reconnection_state
                })
        )
    }

    fn create_data_channel(&self, channel_name: String) -> Result<()> {
        let data_channel = block_on(self.peer_connection.create_data_channel(&channel_name, None))?;
        let webrtc_data_channel = WebRtcDataChannelNative::new(data_channel);
        self.data_channels.lock().insert(channel_name, webrtc_data_channel);  // TODO: Check if this can cause issues
        Ok(())
    }

    fn send_data_to_channel(&self, data: &[u8], channel_name: String) -> Result<()> {
        match self.data_channels.lock().get(&channel_name) {
            Some(channel) => {
                let result = channel.send(data);
                result
            },
            None => {
                Err(anyhow!("Channel name {channel_name} does not exist!"))
            }
        }
    }

    async fn create_offer(&self) -> Result<OfferAnswer> {
        self.local_ice_candidates.lock().clear();  // Empty ice candidates, will gather new ones
        let mut lock = self.caller_type.lock();
        *lock = CallerType::Caller;
        drop(lock);

        let offer_obj = self.peer_connection.create_offer(None).await?;
        let offer_sdp = offer_obj.sdp.clone();
        self.peer_connection.set_local_description(offer_obj).await?;
        self.wait_for_ice_gathering().await;
        let ice_candidates = self.local_ice_candidates.lock().clone();
        let offer = OfferAnswer {
            id: self.id,
            sdp_type: CallerType::Caller,
            sdp: offer_sdp,
            ice_candidates: ice_candidates
        };
        Ok(offer)
    }

    async fn create_answer(&self, offer: OfferAnswer) -> Result<OfferAnswer> {
        self.local_ice_candidates.lock().clear();  // Empty ice candidates, will gather new ones
        let mut lock = self.caller_type.lock();
        *lock = CallerType::Receiver;
        drop(lock);
        *self.remote_id.lock() = Some(offer.id);

        let offer = RTCSessionDescription::offer(offer.sdp)?;
        self.peer_connection.set_remote_description(offer).await?;
        let answer_obj = self.peer_connection.create_answer(None).await?;
        let answer_sdp = answer_obj.sdp.clone();
        self.peer_connection.set_local_description(answer_obj).await?;
        self.wait_for_ice_gathering().await;
        let ice_candidates = self.local_ice_candidates.lock().clone();
        let answer = OfferAnswer {
            id: self.id,
            sdp_type: CallerType::Receiver,
            sdp: answer_sdp,
            ice_candidates: ice_candidates
        };
        Ok(answer)
    }

    async fn receive_answer(&self, answer: OfferAnswer) -> Result<()> {
        *self.remote_id.lock()  = Some(answer.id);
        let answer = RTCSessionDescription::answer(answer.sdp)?;
        self.peer_connection.set_remote_description(answer).await.unwrap();
        Ok(())
    }

    async fn wait_for_ice_gathering(&self) {
        loop {
            match self.peer_connection.ice_gathering_state() {
                RTCIceGatheringState::Complete => {
                    break
                },
                _ => {
                    async_std::task::sleep(std::time::Duration::from_millis(5)).await
                }
            };
        }
    }

    async fn add_remote_ice_candidates(&self, remote_ice_candidates: Vec<IceCandidate>) -> Result<()> {
        for ice_candidate in remote_ice_candidates {
            let candidate = RTCIceCandidateInit {
                candidate: ice_candidate.candidate,
                sdp_mid: Some(ice_candidate.sdp_mid),
                sdp_mline_index: Some(ice_candidate.sdp_m_line_index),
                username_fragment: Some("".to_string())  // TODO: Add user frag to ice candidate
            };
            self.peer_connection.add_ice_candidate(candidate).await.unwrap();
        }
        Ok(())
    }

    fn get_id(&self) -> u128 {
        self.id
    }

	fn get_remote_id(&self) -> Option<u128> {
        self.remote_id.lock().clone()

    }
}

// endregion

// region: Functions

pub fn create_api() -> Result<API> {
    let mut media_engine = MediaEngine::default();
    media_engine.register_default_codecs()?;
    let mut registry = Registry::new();
    registry = register_default_interceptors(registry, &mut media_engine)?;
    let api = APIBuilder::new()
        .with_media_engine(media_engine)
        .with_interceptor_registry(registry)
        .build();
    Ok(api)
}

// endregion


// region: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use ntest::timeout;

    #[tokio::test]
    pub async fn native_webrtc_test() {
        let pc1 = WebRtcConnectionNative::new().unwrap();
        let pc2 = WebRtcConnectionNative::new().unwrap();
        pc1.create_data_channel("data".into()).unwrap();  // NOTE: Always create data channel BEFORE offer and answer!
        let offer = pc1.create_offer().await.unwrap();
        let answer = pc2.create_answer(offer).await.unwrap();
        pc1.receive_answer(answer).await.unwrap();

        // Gather local
        pc1.wait_for_ice_gathering().await;
        pc2.wait_for_ice_gathering().await;
        println!("pc1 ice {:?}", pc1.local_ice_candidates.lock().to_vec());
        println!("pc2 ice {:?}", pc2.local_ice_candidates.lock().to_vec());
        pc2.add_remote_ice_candidates(pc1.local_ice_candidates.lock().to_vec()).await.unwrap();
        pc1.add_remote_ice_candidates(pc2.local_ice_candidates.lock().to_vec()).await.unwrap();
        async_std::task::sleep(std::time::Duration::from_secs(1)).await;
        //println!("pc1 channels: {:?}", pc1.data_channels.lock());
        //println!("pc2 channels: {:?}", pc2.data_channels.lock());

        // PC1 -> PC2
        pc1.send_data_to_channel(&vec![1,2,3], "data".into()).unwrap();
        async_std::task::sleep(std::time::Duration::from_secs(1)).await;
        let msg = pc2.data_channels.lock().get("data").unwrap().receive();
        println!("RECEIVED {:#?}", msg);

        // PC2 -> PC1
        pc2.send_data_to_channel(&vec![4,5,6], "data".into()).unwrap();
        async_std::task::sleep(std::time::Duration::from_secs(1)).await;
        let msg = pc1.data_channels.lock().get("data").unwrap().receive();
        println!("RECEIVED {:#?}", msg);
    }

    #[tokio::test]
    async fn without_string_sdp() {
        let mut m = MediaEngine::default();
        m.register_default_codecs().unwrap();
        let api = APIBuilder::new().with_media_engine(m).build();
        let offer_pc = api.new_peer_connection(RTCConfiguration::default()).await.unwrap();
        let answer_pc = api.new_peer_connection(RTCConfiguration::default()).await.unwrap();
        let _ = offer_pc.create_data_channel("foo", None).await.unwrap();
        let offer = offer_pc.create_offer(None).await.unwrap();
        offer_pc.set_local_description(offer.clone()).await.unwrap();
        answer_pc.set_remote_description(offer).await.unwrap();
        let answer = answer_pc.create_answer(None).await.unwrap();
        answer_pc.set_local_description(answer.clone()).await.unwrap();
        offer_pc.set_remote_description(answer).await.unwrap();
    }

    #[tokio::test]
    async fn with_string_sdp() {
        let mut m = MediaEngine::default();
        m.register_default_codecs().unwrap();
        let api = APIBuilder::new().with_media_engine(m).build();
        let offer_pc = api.new_peer_connection(RTCConfiguration::default()).await.unwrap();
        let answer_pc = api.new_peer_connection(RTCConfiguration::default()).await.unwrap();
        let _ = offer_pc.create_data_channel("foo", None).await.unwrap();
        let offer = offer_pc.create_offer(None).await.unwrap();
        let offer_new = RTCSessionDescription::offer(offer.sdp).unwrap();
        offer_pc.set_local_description(offer_new.clone()).await.unwrap();
        answer_pc.set_remote_description(offer_new).await.unwrap();
        let answer = answer_pc.create_answer(None).await.unwrap();
        let answer_new = RTCSessionDescription::answer(answer.sdp.clone()).unwrap();
        answer_pc.set_local_description(answer_new.clone()).await.unwrap();
        offer_pc.set_remote_description(answer_new.clone()).await.unwrap();
    }

    #[tokio::test]
    #[timeout(10000)]
    //#[ignore]
    async fn create_multiple() {
        let mut pcs = vec![];
        for _ in 0..3 {
            for _ in 0..256 {
                // NOTE: Creating too many at once causes stop! Not sure why, but any wait will solve this
                tokio::time::sleep(std::time::Duration::from_nanos(1)).await;
                let pc = WebRtcConnectionNative::new().unwrap();
                pc.create_data_channel("test".to_string()).unwrap();
                pcs.push(pc);
            }
            println!("LOOP");
            pcs.clear();
        }
    }

    #[tokio::test]
    #[timeout(10000)]
    async fn reuse_api_test() {
        let mut pcs = vec![];
        let api = create_api().unwrap();
        for _ in 0..3 {
            for _ in 0..256 {
                // NOTE: Creating too many at once causes stop! Not sure why, but any wait will solve this
                //tokio::time::sleep(std::time::Duration::from_nanos(1)).await;
                let config = RTCConfiguration::default();
                let pc = api.new_peer_connection(config).await.unwrap();
                pc.create_data_channel("test", None).await.unwrap();
                pcs.push(pc);
            }
            println!("LOOP");
            pcs.clear();
        }
    }

    #[tokio::test]
    #[timeout(10000)]
    async fn new_api_test() {
        let mut pcs = vec![];
        for _ in 0..3 {
            for _ in 0..256 {
                // NOTE: Creating too many at once causes stop! Not sure why, but any wait will solve this
                //tokio::time::sleep(std::time::Duration::from_nanos(1)).await;
                let api = create_api().unwrap();
                let config = RTCConfiguration::default();
                let pc = api.new_peer_connection(config).await.unwrap();
                pc.create_data_channel("test", None).await.unwrap();
                pcs.push(pc);
            }
            println!("LOOP");
            pcs.clear();
        }
    }
}

// endregion