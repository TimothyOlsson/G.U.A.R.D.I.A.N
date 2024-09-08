// std
use std::sync::Arc;

// third-party
use async_trait::async_trait;
use anyhow::Result;
use serde::{Serialize, Deserialize};

#[cfg_attr(
	all(
		target_arch = "wasm32",
		any(target_os = "emscripten", target_os = "unknown"),
	),
	path = "webrtc_wasm.rs"
)]
#[cfg_attr(
	not(
		all(
			target_arch = "wasm32",
			any(target_os = "emscripten", target_os = "unknown"),
		),
	),
	path = "webrtc_native.rs"
)]
pub mod webrtc_connection;

pub const STUN_SERVERS: &[&str] = &[
    //"stun:stun.l.google.com:19302",
    //"stun:stun.nextcloud.com:3478",
    //"stun:151.177.219.79:3478",
];

pub const RECONNECT_SERVER: &str = "http://localhost:8000/reconnect";
pub const RECONNECT_SERVER_RESPONSE_TIMEOUT_S: u64 = 5;
pub const DISCONNECT_WAIT_TIME_S: u64 = 15;


#[cfg(not(target_arch = "wasm32"))]
use webrtc::data_channel::RTCDataChannel;
#[cfg(not(target_arch = "wasm32"))]
type WebRtcDataChannelType = RTCDataChannel;

#[cfg(target_arch = "wasm32")]
use web_sys::RtcDataChannel;
#[cfg(target_arch = "wasm32")]
type WebRtcDataChannelType = RtcDataChannel;


pub trait WebRtcConnection {
    fn new() -> Result<Box<Self>>;
    fn create_data_channel(&self, channel_name: String) -> Result<()>;
    fn send_data_to_channel(&self, data: &[u8], channel_name: String) -> Result<()>;
    async fn create_offer(&self) -> Result<OfferAnswer>;
    async fn create_answer(&self, offer: OfferAnswer) -> Result<OfferAnswer>;
	async fn receive_answer(&self, answer_sdp: OfferAnswer) -> Result<()>;
    async fn wait_for_ice_gathering(&self);
    async fn add_remote_ice_candidates(&self, remote_ice_candidates: Vec<IceCandidate>) -> Result<()>;
	fn get_id(&self) -> u128;
	fn get_remote_id(&self) -> Option<u128>;

}

pub trait WebRtcDataChannel {
    fn new(data_channel: Arc<WebRtcDataChannelType>) -> Self;
    fn send(&self, data: &[u8]) -> Result<()>;
    fn receive(&self) -> Vec<Vec<u8>>;
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IceCandidate {
	candidate: String,
	sdp_mid: String,
	sdp_m_line_index: u16,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OfferAnswer {
	id: u128,
	sdp_type: CallerType,  // Caller = offer, Receiver = answer
	sdp: String,
	ice_candidates: Vec<IceCandidate>
}

#[derive(Debug, Clone)]
pub struct ReconnectionState {
	reconnection_initiated: bool,
	reconnection_failed: bool
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum CallerType {
	Receiver,
	Caller,
}

/// Needed to handle wasm and native the same
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum RtcPeerConnectionState {
	New,
	Connecting,
	Connected,
	Disconnected,
	Failed,
	Closed
}


