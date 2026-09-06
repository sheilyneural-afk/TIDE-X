#![no_main]

use cerebro_tidex::capability_bundle::CapabilityBundle;
use cerebro_tidex::capability_ir::CapabilityIr;
use cerebro_tidex::content_vault::CaptureReceipt;
use cerebro_tidex::procedural_memory::{DriftRecord, SolverAttempt, SolverRunFailureRecord};
use libfuzzer_sys::fuzz_target;

fn round_trip<T>(bytes: &[u8])
where
    T: serde::de::DeserializeOwned + serde::Serialize,
{
    if let Ok(value) = serde_json::from_slice::<T>(bytes) {
        let encoded = serde_json::to_vec(&value).expect("accepted value must serialize");
        let _: T = serde_json::from_slice(&encoded)
            .expect("serialized accepted value must deserialize");
    }
}

fuzz_target!(|bytes: &[u8]| {
    round_trip::<CaptureReceipt>(bytes);
    round_trip::<CapabilityIr>(bytes);
    round_trip::<CapabilityBundle>(bytes);
    round_trip::<SolverAttempt>(bytes);
    round_trip::<DriftRecord>(bytes);
    round_trip::<SolverRunFailureRecord>(bytes);
});
