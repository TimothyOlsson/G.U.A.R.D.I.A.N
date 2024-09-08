///! TODO: Add documentation

use std::{
    sync::Arc,
    collections::HashMap,
    any::Any,
    time::Duration
};

// wasm
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use js_sys::{Reflect, Array, Object, Uint8Array};
use web_sys::{
    RtcDataChannelEvent, RtcPeerConnection, RtcPeerConnectionIceEvent, RtcSdpType,
    RtcSessionDescriptionInit, RtcConfiguration, RtcDataChannel, RtcIceGatheringState,
    MessageEvent, RtcIceCandidateInit, RtcIceCandidate, Blob, RtcIceConnectionState,
    RtcDataChannelState, RtcOfferOptions, RequestInit, Request, RequestMode, Response
};

// Other
use anyhow::{anyhow, Result, Context};
use parking_lot::Mutex;
use rand::Rng;
use crossbeam::channel::Receiver;
use crossbeam::channel::unbounded;
use async_trait::async_trait;
use tracing::{debug, warn};
use async_std::future;
use hex;

// Local crates
use crate::WebRtcConnection;
use crate::WebRtcDataChannel;
use crate::WebRtcDataChannelType;
use crate::IceCandidate;
use crate::CallerType;
use crate::OfferAnswer;
use crate::ReconnectionState;
use crate::RECONNECT_SERVER;
use crate::RECONNECT_SERVER_RESPONSE_TIMEOUT_S;
use crate::STUN_SERVERS;
use crate::DISCONNECT_WAIT_TIME_S;

// region: Types

pub type WebRtcGuardian = WebRtcConnectionWasm;

// endregion

// region: Structs

#[allow(dead_code)]
#[derive(Clone)]
pub struct WebRtcConnectionWasm {
    id: u128,
    remote_id: Arc<Mutex<Option<u128>>>,
    caller_type: Arc<Mutex<CallerType>>,
    peer_connection: Arc<RtcPeerConnection>,
    local_ice_candidates: Arc<Mutex<Vec<IceCandidate>>>,
    data_channels: Arc<Mutex<HashMap<String, WebRtcDataChannelWasm>>>,
    reconnection_state: Arc<Mutex<ReconnectionState>>,
    _callbacks: Option<Arc<Mutex<Vec<Box<dyn Any>>>>>
}

#[derive(Debug)]
pub struct WebRtcDataChannelWasm {
    data_channel: Arc<RtcDataChannel>,
    msg_received: Receiver<Vec<u8>>,
    _callbacks: Vec<Box<dyn Any>>
}

// endregion

// region: Enums

enum Method {
    GET,
    POST
}

// endregion

// region: Implementations

impl WebRtcDataChannel for WebRtcDataChannelWasm {

    fn new(data_channel: Arc<WebRtcDataChannelType>) -> Self {

        // Store callbacks
        let mut _callbacks: Vec<Box<dyn Any>> = vec![];

        // Channels
        let (msg_incoming, msg_received) = unbounded::<Vec<u8>>();
        let msg_incoming_clone = msg_incoming.clone();

        // What to do when receiving a message from data_channel
        let onmessage_callback =
        Closure::wrap(
            Box::new(move |ev: MessageEvent| {
                if let Ok(abuf) = ev.data().dyn_into::<js_sys::ArrayBuffer>() {  // Chrome sends ArrayBuffer
                    debug!("Received ArrayBuffer!");
                    let array = Uint8Array::new(&abuf);
                    let msg = array.to_vec();
                    msg_incoming_clone.try_send(msg).unwrap();  // Could this fail?
                } else if let Ok(blob) = ev.data().dyn_into::<Blob>() {  // Firefox sends Blob
                    debug!("Received blob!");
                    let fr = web_sys::FileReader::new().unwrap();
                    let fr_c = fr.clone();
                    let msg_incoming_clone2 = msg_incoming.clone();
                    let onloadend_cb =
                    Closure::wrap(Box::new(move |_e: web_sys::ProgressEvent| {
                        let array = js_sys::Uint8Array::new(&fr_c.result().unwrap());
                        let msg = array.to_vec();
                        msg_incoming_clone2.try_send(msg).unwrap();    // Could this fail?
                    }) as Box<dyn FnMut(web_sys::ProgressEvent)>);
                    fr.set_onloadend(Some(onloadend_cb.as_ref().unchecked_ref()));
                    fr.read_as_array_buffer(&blob).expect("blob not readable");
                    onloadend_cb.forget();
                } else if let Ok(_txt) = ev.data().dyn_into::<js_sys::JsString>() {
                    debug!("Received text!");
                } else {
                    debug!("Received unknown data!");
                }
            }) as Box<dyn FnMut(MessageEvent)>);
        data_channel.set_onmessage(Some(onmessage_callback.as_ref().unchecked_ref()));
        _callbacks.push(Box::new(onmessage_callback));

        // What to do when new datachannel is open
        let onopen_callback =
        Closure::wrap(
            Box::new(move |_ev: RtcDataChannelEvent| {

            }) as Box<dyn FnMut(RtcDataChannelEvent)>
        );
        data_channel.set_onopen(Some(onopen_callback.as_ref().unchecked_ref()));
        _callbacks.push(Box::new(onopen_callback));

        Self {
            data_channel,
            msg_received,
            _callbacks
        }
    }

    fn send(&self, data: &[u8]) -> Result<()> {
        // NOTE: send_with_u8_array does not work with SharedArrayBuffer enabled, since it defaults creating &[u8] to SharedArrayBuffer
        // instead of ArrayBuffer. SharedArrayBuffer is also not an transferable object, causing errors with webrtc
        // (https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer)
        // Thus, doing a manual array to send is better for portability, if SharedArrayBuffer is enabled
        match self.data_channel.ready_state() {
            RtcDataChannelState::Open => {},  // will continue to send
            state => {
                return Err(anyhow!("Channel state is {state:?}. Cannot send data!"));
            }
        }
        let array = Uint8Array::from(data).buffer();
        let sent = self.data_channel.send_with_array_buffer(&array);
        js_error_wrapper(sent)
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
impl WebRtcConnection for WebRtcConnectionWasm {

    fn new() -> Result<Box<Self>> {
        // Create ID
        let mut rng = rand::thread_rng();
        let id: u128 = rng.gen(); 
        let remote_id = Arc::new(Mutex::new(None));

        // Data to fill
        let caller_type = Arc::new(Mutex::new(CallerType::Receiver));  // Always default to "receiver"
        let local_ice_candidates = Arc::new(Mutex::new(Vec::<IceCandidate>::new()));
        let data_channels = Arc::new(Mutex::new(HashMap::<String, WebRtcDataChannelWasm>::new()));
        let reconnection_state = Arc::new(Mutex::new(ReconnectionState { reconnection_initiated: false, reconnection_failed: false}));
        let mut _callbacks: Arc<Mutex<Vec<Box<dyn Any>>>> = Arc::new(Mutex::new(vec![]));

        // Create config
        let mut config = RtcConfiguration::new();
        let ice_servers = Array::new();
        {
            for stun_server in STUN_SERVERS {
                let server_entry = Object::new();
                js_error_wrapper(
                    Reflect::set(&server_entry,
                                 &"urls".into(),
                                 &stun_server.to_string().into())
                )?;
                ice_servers.push(&*server_entry);
            }
        }
        debug!("{:?}", ice_servers);
        config.ice_servers(&ice_servers);

        // Create connection
        let peer_connection = Arc::new(
            js_error_wrapper(
                RtcPeerConnection::new_with_configuration(&config)
            )?
        );
        debug!("PC {id} created: state {:?}", peer_connection.signaling_state());

        // Set method for gathering ICE candidates
        let local_ice_candidates_clone = local_ice_candidates.clone();
        let onicecandidate_callback =
        Closure::wrap(
            Box::new(move |ev: RtcPeerConnectionIceEvent| match ev.candidate() {
                Some(candidate) => {
                    debug!("Candidate: {:?}", candidate.to_json());
                    let ice_candidate = IceCandidate {
                        candidate: candidate.candidate(),
                        sdp_mid: candidate.sdp_mid().unwrap_or("0".to_string()),
                        sdp_m_line_index: candidate.sdp_m_line_index().unwrap_or(0)
                    };
                    debug!("PC {id}, onicecandidate: {:#?}", ice_candidate);
                    local_ice_candidates_clone.lock().push(ice_candidate);
                }
                None => {}
            }) as Box<dyn FnMut(RtcPeerConnectionIceEvent)>,
        );
        peer_connection.set_onicecandidate(Some(onicecandidate_callback.as_ref().unchecked_ref()));
        _callbacks.lock().push(Box::new(onicecandidate_callback));

        // A method to save received data channels and messages
        let data_channels_clone = data_channels.clone();
        let ondatachannel_callback =
        Closure::wrap(
            Box::new(move |ev: RtcDataChannelEvent| {
                let data_channel = Arc::new(ev.channel());
                let data_channel_label = data_channel.label();
                debug!("PC {id}, received channel: {data_channel_label}");

                // Init datachannel
                let webrtc_data_channel = WebRtcDataChannelWasm::new(data_channel);

                // Move data channel into hashmap
                data_channels_clone.lock().insert(data_channel_label, webrtc_data_channel);

            }) as Box<dyn FnMut(RtcDataChannelEvent)>);
        peer_connection.set_ondatachannel(Some(ondatachannel_callback.as_ref().unchecked_ref()));
        _callbacks.lock().push(Box::new(ondatachannel_callback));


        // To use when adding the last callback for reconnection
        let _callbacks_clone = _callbacks.clone();

        let webrtc_connection =
        Box::new(
            Self {
                id,
                remote_id,
                caller_type,
                peer_connection,
                local_ice_candidates,
                data_channels,
                reconnection_state,
                _callbacks: Some(_callbacks)
            }
        );

        // What to do during a reconnect
        // Callback-hell... http://callbackhell.com/
        let peer_connection_clone = webrtc_connection.peer_connection.clone();
        let peer_connection_clone2 = webrtc_connection.peer_connection.clone();
        let mut webrtc_connection_clone = webrtc_connection.clone();
        webrtc_connection_clone._callbacks = None;  // Remove reference to itself, prevent circular reference
        // Not sure if the above fix works...
        // Since "on_ice_connection_state_change" exists in _callbacks,
        // whereas the callback itself contains the struct with a Arc pointing to itself
        // Will this lead to a memory leak? I have no idea.
        let on_ice_connection_state_change =
        Closure::wrap(
            Box::new(move |_state: RtcIceConnectionState| {
                debug!("STATE {:?} for {id}", peer_connection_clone.ice_connection_state());
                match peer_connection_clone.ice_connection_state() {
                    RtcIceConnectionState::Disconnected | RtcIceConnectionState::Failed => {
                        warn!("PC {id} connection disconnected!");
                        // Required to be cloned again, to move into
                        // This is quite messy...
                        let webrtc_connection_clone2 = webrtc_connection_clone.clone();
                        wasm_bindgen_futures::spawn_local(
                            reconnect_wrapper(*webrtc_connection_clone2,
                                              RECONNECT_SERVER)
                        );
                    },
                    _ => {}  // Ignore rest
                };
            }) as Box<dyn FnMut(RtcIceConnectionState)>);
        peer_connection_clone2.set_oniceconnectionstatechange(Some(on_ice_connection_state_change.as_ref().unchecked_ref()));
        _callbacks_clone.lock().push(Box::new(on_ice_connection_state_change));

        // TODO: Handle removal of channels, if it exists

        Ok(webrtc_connection)
    }

    fn create_data_channel(&self, channel_name: String) -> Result<()> {
        if self.data_channels.lock().contains_key(&channel_name) {
            return Err(anyhow!("Channel already exists!"));
        }
        let data_channel = Arc::new(self.peer_connection.create_data_channel(&channel_name));
        let webrtc_data_channel = WebRtcDataChannel::new(data_channel);
        self.data_channels.lock().insert(channel_name, webrtc_data_channel);  // TODO: Check if this can cause issues,
        Ok(())
    }

    fn send_data_to_channel(&self, data: &[u8], channel_name: String) -> Result<()> {
        match self.peer_connection.ice_connection_state() {
            RtcIceConnectionState::Connected => {},
            RtcIceConnectionState::Completed => {},  // Is this ok?
            state => {
                return Err(anyhow!("The PC state is {state:?}. Data cannot be sent!"));
            }
        }
        match self.data_channels.lock().get(&channel_name) {
            Some(channel) => {
                let result = channel.send(data);
                // If fail, consider ice-restart?
                result
            },
            None => {
                Err(anyhow!("Channel name {channel_name} does not exist!"))
            }
        }
    }

    async fn create_offer(&self) -> Result<OfferAnswer> {
        // Clean up
        self.local_ice_candidates.lock().clear();  // Empty ice candidates, will gather new ones
        let data_channel_names: Vec<String> = self.data_channels.lock().keys().cloned().collect();
        self.data_channels.lock().clear();  // Empty data channels, new ones will be created
        let _ = self.create_data_channel("data".to_string());  // Create default channel, if no channel has been created before creating an offer
        for data_channel_name in data_channel_names {
            let _ = self.create_data_channel(data_channel_name);  // Fill in previous channels
        }
        let mut lock = self.caller_type.lock();
        *lock = CallerType::Caller;
        drop(lock);

        // Start
        let mut rtc_options = RtcOfferOptions::new();
        rtc_options.ice_restart(true);
        let offer = js_error_wrapper(
            JsFuture::from(self.peer_connection.create_offer_with_rtc_offer_options(&rtc_options)).await
        )?;
        let offer_sdp = js_error_wrapper(
            Reflect::get(&offer, &JsValue::from_str("sdp"))
        )?.as_string()
            .unwrap();
        debug!("PC {}: Offer {:?}", self.id, offer_sdp);
        let mut offer_obj = RtcSessionDescriptionInit::new(RtcSdpType::Offer);
        offer_obj.sdp(&offer_sdp);
        let sld_promise = self.peer_connection.set_local_description(&offer_obj);
        js_error_wrapper(
            JsFuture::from(sld_promise).await
        )?;
        debug!("PC {}: State {:?}", self.id, self.peer_connection.signaling_state());
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
        // Clean
        self.data_channels.lock().clear();  // Empty datachannels, new ones will be created
        self.local_ice_candidates.lock().clear();  // Empty ice candidates to gather new ones
        let mut lock = self.caller_type.lock();
        *lock = CallerType::Receiver;
        drop(lock);

        // Start
        *self.remote_id.lock() = Some(offer.id);
        let mut offer_obj = RtcSessionDescriptionInit::new(RtcSdpType::Offer);
        offer_obj.sdp(&offer.sdp);
        let srd_promise = self.peer_connection.set_remote_description(&offer_obj);
        js_error_wrapper(
            JsFuture::from(srd_promise).await
        )?;
        debug!("PC {}: State {:?}", self.id, self.peer_connection.signaling_state());
        let answer = js_error_wrapper(
            JsFuture::from(self.peer_connection.create_answer()).await
        )?;
        let answer_sdp = js_error_wrapper(
            Reflect::get(&answer, &JsValue::from_str("sdp"))
        )?.as_string().unwrap();
        debug!("PC {}: answer {:?}", self.id, answer_sdp);
        let mut answer_obj = RtcSessionDescriptionInit::new(RtcSdpType::Answer);
        answer_obj.sdp(&answer_sdp);
        let sld_promise = self.peer_connection.set_local_description(&answer_obj);
        js_error_wrapper(
            JsFuture::from(sld_promise).await
        )?;
        self.add_remote_ice_candidates(offer.ice_candidates).await?;
        debug!("PC {}: State {:?}", self.id, self.peer_connection.signaling_state());
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
        let mut answer_obj = RtcSessionDescriptionInit::new(RtcSdpType::Answer);
        answer_obj.sdp(&answer.sdp);
        let srd_promise = self.peer_connection.set_remote_description(&answer_obj);
        js_error_wrapper(
            JsFuture::from(srd_promise).await
        )?;
        debug!("pc: state {:?}", self.peer_connection.signaling_state());
        self.add_remote_ice_candidates(answer.ice_candidates).await?;
        Ok(())
    }

    /// NOTE: This can take up to 20 seconds...
    /// However, trickle ICE candidates is much more difficult to sync
    /// https://stackoverflow.com/questions/34217462/webrtc-connects-on-local-connection-but-fails-over-internet
    async fn wait_for_ice_gathering(&self) {
        loop {
            match self.peer_connection.ice_gathering_state() {
                RtcIceGatheringState::Complete => {
                    debug!("Ice gathering complete");
                    break
                },
                state => {
                    debug!("Waiting... State {state:?}");
                    async_std::task::sleep(std::time::Duration::from_millis(10)).await
                }
            };
        }
    }

    async fn add_remote_ice_candidates(&self, remote_ice_candidates: Vec<IceCandidate>) -> Result<()> {
        for ice_candidate in remote_ice_candidates {
            let mut ice_candidate_init = RtcIceCandidateInit::new(&ice_candidate.candidate);
            ice_candidate_init.sdp_mid(Some(&ice_candidate.sdp_mid));
            ice_candidate_init.sdp_m_line_index(Some(ice_candidate.sdp_m_line_index));
            let ice_candidate_obj = 
            js_error_wrapper(
                RtcIceCandidate::new(&ice_candidate_init)
            )?;
            let promise = self.peer_connection.add_ice_candidate_with_opt_rtc_ice_candidate(Some(&ice_candidate_obj));
            let _ = wasm_bindgen_futures::JsFuture::from(promise).await.unwrap();
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

impl Drop for WebRtcConnectionWasm {
    fn drop(&mut self) {
        let ref_count = Arc::strong_count(&self.peer_connection);
        // If I understand, if it is one, it means this is the last reference to the peer connection
        // In that case, close peer connection
        if ref_count <= 1 { 
            self.peer_connection.close();
        }
    }
}

// endregion

// region: Functions

pub fn js_error_wrapper<T>(value: Result<T, JsValue>) -> Result<T> {
    match value {
        Ok(value) => {Ok(value)},
        Err(e) => {
            let msg = e;
            Err(anyhow!("JS error: {msg:?}"))
        }
    }
}

async fn get_response(url: &str, method: Method, timeout_time: u64) -> Result<Response> {
    let mut opts = RequestInit::new();
    let method = match method {
        Method::GET => {"GET"},
        Method::POST => {"POST"}
    };
    opts.method(method);
    opts.mode(RequestMode::Cors);
    let request = Request::new_with_str_and_init(&url, &opts).unwrap();  // Could this fail?
    let window = web_sys::window().unwrap();  // Could this fail?
    let resp_value_future = JsFuture::from(window.fetch_with_request(&request));
    // JS prints 404 errors here, not sure how to stop it...
    let timeout = future::timeout(Duration::from_secs(timeout_time), resp_value_future).await?;
    let resp_value = js_error_wrapper(timeout)?;
    let resp: Response = js_error_wrapper(resp_value.dyn_into())?;
    Ok(resp)
}

async fn get_text_from_response(resp: Response) -> Result<String> {
    if !resp.ok() {
        return Err(anyhow!("Response is not ok! {resp:?}"));
    }
    let txt: String = js_error_wrapper(
                            JsFuture::from(
                                js_error_wrapper(resp.text())?
                            ).await
                      )?
                      .as_string()
                      .context("Not a string")?;
    Ok(txt)
}

fn serialize_and_encode(offer_answer: OfferAnswer) -> Result<String> {
    let offer_answer_serialized = bincode::serialize(&offer_answer)?;
    let offer_answer_encoded = hex::encode(offer_answer_serialized);
    Ok(offer_answer_encoded)
}

fn decode_and_deserialize(offer_answer_encoded: String) -> Result<OfferAnswer> {
    let offer_answer_serialized = hex::decode(&offer_answer_encoded)?;
    let offer_answer: OfferAnswer = bincode::deserialize(&offer_answer_serialized)?;
    Ok(offer_answer)
}

/// NOTE: Requires ownership, due to wasm_bindgen_futures::spawn_local
pub async fn reconnect_wrapper(
    webrtc_connection: WebRtcConnectionWasm,
    signaling_server_url: &str
) {
    if webrtc_connection.reconnection_state.lock().reconnection_initiated {
        return;
    }
    webrtc_connection.reconnection_state.lock().reconnection_initiated = true;  // Lock reconnection
    match webrtc_connection.peer_connection.ice_connection_state() {
        RtcIceConnectionState::Failed => { },  // Reconnect immediately
        RtcIceConnectionState::Disconnected => {  // Wait X seconds
            // Wait X
            let n_loops = DISCONNECT_WAIT_TIME_S * 1000 / 10;
            for _ in 0..n_loops {
                match webrtc_connection.peer_connection.ice_connection_state() {
                    RtcIceConnectionState::Failed => { break },  // Reconnect immediately
                    RtcIceConnectionState::Disconnected => { },  // Keep waiting
                    _ => { 
                        webrtc_connection.reconnection_state.lock().reconnection_initiated = false;  // Unlock reconnection
                        return;
                    }
                };
                async_std::task::sleep(std::time::Duration::from_millis(10)).await
            }
        },
        _ => { 
            // Could be closed here
            webrtc_connection.reconnection_state.lock().reconnection_initiated = false;  // Unlock reconnection
            return;
        }
    }
    let id = webrtc_connection.id;
    let result = reconnect(&webrtc_connection,
                           signaling_server_url).await;
    match result {
        Ok(_) => { },
        Err(e) => {
            debug!("PC {id} failed to reconnect. Error {e}");
            webrtc_connection.reconnection_state.lock().reconnection_failed = true;
        }
    }
    debug!("PC {id} reconnected");
    webrtc_connection.reconnection_state.lock().reconnection_initiated = false;  // Unlock reconnection
}

async fn reconnect(
    webrtc_connection: &WebRtcConnectionWasm,
    signaling_server_url: &str
) -> Result<()> {
    let caller_type_clone = webrtc_connection.caller_type.lock().clone();
    match caller_type_clone {
        CallerType::Caller => {
            let id = webrtc_connection.id;
            let offer = webrtc_connection.create_offer().await?;
            let offer_encoded = serialize_and_encode(offer)?;
            let url = format!(r#"{signaling_server_url}/set_offer/{id}/{offer_encoded}"#);
            warn!("{url}");
            // Try to send 5 times, then exit if not working
            let mut offer_sent = false;
            for _ in 0..5 {
                let resp = get_response(&url, Method::POST, RECONNECT_SERVER_RESPONSE_TIMEOUT_S).await;
                match resp {
                    Ok(resp) => {
                        if resp.ok() {
                            offer_sent = true;
                            break;
                        }
                    },
                    Err(_) => {}
                };
                async_std::task::sleep(Duration::from_secs(5)).await;
            }
            if !offer_sent {
                return Err(anyhow!("Could not send offer to signaling server!"));
            }

            let url = format!(r#"{signaling_server_url}/get_answer/{id}"#);
            warn!("{url}");

            // Check every 2 seconds for an answer, for 30 seconds
            let mut answer: Option<OfferAnswer> = None;
            for _ in 0..15 {
                async_std::task::sleep(Duration::from_secs(2)).await;
                let resp = get_response(&url, Method::GET, RECONNECT_SERVER_RESPONSE_TIMEOUT_S).await;
                match resp {
                    Ok(resp) => {
                        if resp.ok() {
                            warn!("Received! {resp:?}");
                            let answer_encoded: String = get_text_from_response(resp).await?;
                            warn!("Answer! {:?}", answer_encoded);
                            answer = Some(decode_and_deserialize(answer_encoded)?);
                            warn!("Answer! {:?}", answer);
                            break;
                        }
                    },
                    Err(_) => {}
                };
            }
            if answer.is_none() {
                return Err(anyhow!("Answer not received from signaling server!"));
            }
            webrtc_connection.receive_answer(answer.unwrap()).await.unwrap();
            warn!("Ok");
            return Ok(());
        },
        CallerType::Receiver => {
            let remote_id = webrtc_connection.remote_id.lock().unwrap();  // Should never fail here
            let url = format!(r#"{signaling_server_url}/get_offer/{remote_id}"#);
            let mut offer: Option<OfferAnswer> = None;
            for _ in 0..15 {
                async_std::task::sleep(Duration::from_secs(2)).await;
                let resp = get_response(&url, Method::GET, RECONNECT_SERVER_RESPONSE_TIMEOUT_S).await;
                match resp {
                    Ok(resp) => {
                        if resp.ok() {
                            warn!("Received! {resp:?}");
                            let offer_encoded: String = get_text_from_response(resp).await?;
                            warn!("Offer! {:?}", offer_encoded);
                            offer = Some(decode_and_deserialize(offer_encoded)?);
                            warn!("Offer! {:?}", offer);
                            break;
                        }
                    },
                    Err(_) => {}
                };
            }
            if offer.is_none() {
                return Err(anyhow!("Offer not received from signaling server!"));
            }
            let answer = webrtc_connection.create_answer(offer.unwrap()).await?;
            let answer_encoded = serialize_and_encode(answer)?;
            let url = format!(r#"{signaling_server_url}/set_answer/{remote_id}/{answer_encoded}"#);
            warn!("{url}");
            let mut answer_sent = false;
            for _ in 0..5 {
                let resp = get_response(&url, Method::POST, RECONNECT_SERVER_RESPONSE_TIMEOUT_S).await;
                match resp {
                    Ok(resp) => {
                        if resp.ok() {
                            answer_sent = true;
                            break;
                        }
                    },
                    Err(_) => {}
                };
                async_std::task::sleep(Duration::from_secs(5)).await;
            }
            if !answer_sent {
                return Err(anyhow!("Could not send answer to signaling server!"));
            }
            return Ok(());
        }
    };
}

// endregion

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;
    use tracing::info;
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    pub async fn wasm_webrtc_test() {
        let pc1 = WebRtcConnectionWasm::new().unwrap();
        let pc2 = WebRtcConnectionWasm::new().unwrap();
        info!("Creating channel");
        pc1.create_data_channel("data".into()).unwrap();
        info!("Channel created");
        info!("Creating offer");
        let offer = pc1.create_offer().await.unwrap();
        info!("Creating answer");
        let answer = pc2.create_answer(offer).await.unwrap();
        info!("Receiving answer");
        pc1.receive_answer(answer).await.unwrap();

        // Some time between connection and sending data is required. About 100ms
        async_std::task::sleep(Duration::from_millis(100)).await;

        // PC1 -> PC2
        let msg_sent = vec![1,2,3];
        pc1.send_data_to_channel(&msg_sent, "data".into()).unwrap();
        async_std::task::sleep(Duration::from_secs(1)).await;
        let msg_recv = pc2.data_channels.lock().get("data").unwrap().receive();
        info!("RECEIVED {:#?}", msg_recv);
        assert_eq!(msg_sent, *msg_recv.first().unwrap());

        // PC2 -> PC1
        let msg_sent = vec![4,5,6];
        pc2.send_data_to_channel(&msg_sent, "data".into()).unwrap();
        async_std::task::sleep(Duration::from_secs(1)).await;
        let msg_recv = pc1.data_channels.lock().get("data").unwrap().receive();
        info!("RECEIVED {:#?}", msg_recv);
        assert_eq!(msg_sent, *msg_recv.first().unwrap());
    }

    #[wasm_bindgen_test]
    #[ignore]
    pub async fn close_datachannel_test() {
        unimplemented!();  // Should we add this?
    }

    #[wasm_bindgen_test]
    async fn create_multiple() {
        let mut pcs = vec![];
        for _ in 0..256 {
            let pc = WebRtcConnectionWasm::new().unwrap();
            pc.create_data_channel("test".to_string());
            pcs.push(pc);
        }
    }

    #[wasm_bindgen_test]
    async fn reconnect_test() {
        let pc = WebRtcConnectionWasm::new().unwrap();
        let _offer = pc.create_offer().await.unwrap();
    }

    #[wasm_bindgen_test]
    async fn reconnect_test_wrong_id() {

    }
}

pub async fn reconnect_test_receiver_disconnected() {
    let pc1 = WebRtcConnectionWasm::new().unwrap();
    let pc2 = WebRtcConnectionWasm::new().unwrap();
    let offer = pc1.create_offer().await.unwrap();
    let answer = pc2.create_answer(offer).await.unwrap();
    pc1.receive_answer(answer).await.unwrap();
    async_std::task::sleep(Duration::from_millis(200)).await;

    // PC1 -> PC2
    let msg_sent = vec![1,2,3];
    pc1.send_data_to_channel(&msg_sent, "data".into()).unwrap();
    async_std::task::sleep(Duration::from_secs(1)).await;
    let msg_recv = pc2.data_channels.lock().get("data").unwrap().receive();
    assert_eq!(msg_sent, *msg_recv.first().unwrap());

    warn!("Testing reconnection");
    let pc2_id = pc2.id;
    drop(pc2);  // Drop to remove it
    async_std::task::sleep(Duration::from_secs(20)).await;  // Wait for it to recognise that it has been dropped

    // Create a new connection with the same parameters
    let mut pc2 = WebRtcConnectionWasm::new().unwrap();
    pc2.id = pc2_id;
    async_std::task::sleep(Duration::from_secs(15)).await;
    let url = format!("http://localhost:8000/reconnect/get_offer/{}", pc1.id);
    let resp = get_response(&url, Method::GET, 5).await.unwrap();
    let txt = get_text_from_response(resp).await.unwrap();
    warn!("{txt}");
    let offer = decode_and_deserialize(txt).unwrap();
    let answer = pc2.create_answer(offer).await.unwrap();
    let answer_encoded = serialize_and_encode(answer).unwrap();
    let url = format!("http://localhost:8000/reconnect/set_answer/{}/{}", pc1.id, answer_encoded);
    let _ = get_response(&url, Method::POST, 5).await.unwrap();
    async_std::task::sleep(Duration::from_secs(20)).await;
    let reconnected = pc1.reconnection_state.lock().clone();
    warn!("PC1: Failed to reconnect? {reconnected:?}");

    warn!("pc1 {:?}", pc1.peer_connection.ice_connection_state());
    warn!("pc2 {:?}", pc2.peer_connection.ice_connection_state());

    // PC1 -> PC2
    let msg_sent = vec![1,2,3];
    pc1.send_data_to_channel(&msg_sent, "data".into()).unwrap();
    async_std::task::sleep(Duration::from_secs(1)).await;
    let msg_recv = pc2.data_channels.lock().get("data").unwrap().receive();
    assert_eq!(msg_sent, *msg_recv.first().unwrap());
}


pub async fn reconnect_test_caller_disconnected() {
    let pc1 = WebRtcConnectionWasm::new().unwrap();
    let pc2 = WebRtcConnectionWasm::new().unwrap();
    let offer = pc1.create_offer().await.unwrap();
    let answer = pc2.create_answer(offer).await.unwrap();
    pc1.receive_answer(answer).await.unwrap();
    async_std::task::sleep(Duration::from_millis(200)).await;

    // PC1 -> PC2
    let msg_sent = vec![1,2,3];
    pc1.send_data_to_channel(&msg_sent, "data".into()).unwrap();
    async_std::task::sleep(Duration::from_secs(1)).await;
    let msg_recv = pc2.data_channels.lock().get("data").unwrap().receive();
    assert_eq!(msg_sent, *msg_recv.first().unwrap());

    // New data channel
    pc1.create_data_channel("hello".to_string()).unwrap();
    async_std::task::sleep(Duration::from_millis(200)).await;
    warn!("pc1: {:?}", pc1.data_channels);
    warn!("pc2: {:?}", pc2.data_channels);

    pc2.create_data_channel("hello2".to_string()).unwrap();
    async_std::task::sleep(Duration::from_millis(200)).await;
    warn!("pc1: {:?}", pc1.data_channels);
    warn!("pc2: {:?}", pc2.data_channels);

    warn!("Testing reconnection");
    let pc1_id = pc1.id;
    drop(pc1);  // Drop to remove it
    async_std::task::sleep(Duration::from_secs(20)).await;  // Wait for it to recognise that it has been dropped

    // Create a new connection with the same parameters
    let mut pc1 = WebRtcConnectionWasm::new().unwrap();
    pc1.id = pc1_id;
    let offer = pc1.create_offer().await.unwrap();
    let offer_encoded = serialize_and_encode(offer).unwrap();
    let url = format!("http://localhost:8000/reconnect/set_offer/{}/{}", pc1.id, offer_encoded);
    let _ = get_response(&url, Method::POST, 5).await.unwrap();

    async_std::task::sleep(Duration::from_secs(20)).await;
    let url = format!("http://localhost:8000/reconnect/get_answer/{}", pc1.id);
    let resp = get_response(&url, Method::GET, 5).await.unwrap();
    let txt = get_text_from_response(resp).await.unwrap();
    warn!("Response: {txt}");
    let answer = decode_and_deserialize(txt).unwrap();
    warn!("Answer: {answer:?}");
    pc1.receive_answer(answer).await.unwrap();

    async_std::task::sleep(Duration::from_secs(20)).await;

    let reconnected = pc2.reconnection_state.lock().clone();
    warn!("PC2: Failed to reconnect? {reconnected:?}");

    warn!("pc1 {:?}", pc1.peer_connection.ice_connection_state());
    warn!("pc2 {:?}", pc2.peer_connection.ice_connection_state());

    warn!("pc1 {:?}", pc1.data_channels);
    warn!("pc2 {:?}", pc2.data_channels);

    // PC1 -> PC2
    let msg_sent = vec![1,2,3];
    pc1.send_data_to_channel(&msg_sent, "data".into()).unwrap();
    async_std::task::sleep(Duration::from_secs(1)).await;
    let msg_recv = pc2.data_channels.lock().get("data").unwrap().receive();
    assert_eq!(msg_sent, *msg_recv.first().unwrap());
}

// endregion