//! Exact, finite floating-point values for authenticated wire formats.

use crate::error::{BrainError, BrainResult};
use serde::{de::Error as DeError, Deserialize, Deserializer, Serialize, Serializer};

macro_rules! finite_float {
    ($name:ident, $float:ty, $bits:ty, $width:expr, $label:literal) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name($bits);

        impl $name {
            pub fn new(value: $float) -> BrainResult<Self> {
                if !value.is_finite() {
                    return Err(BrainError::Numerical(concat!($label, "_non_finite").into()));
                }
                Ok(Self(value.to_bits()))
            }

            pub fn get(self) -> $float {
                <$float>::from_bits(self.0)
            }

            pub fn bits(self) -> $bits {
                self.0
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.serialize_str(&format!(concat!("{:0", $width, "x}"), self.0))
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let encoded = String::deserialize(deserializer)?;
                if encoded.len() != $width
                    || encoded
                        .bytes()
                        .any(|byte| !byte.is_ascii_hexdigit() || byte.is_ascii_uppercase())
                {
                    return Err(D::Error::custom(concat!($label, "_encoding_invalid")));
                }
                let bits = <$bits>::from_str_radix(&encoded, 16)
                    .map_err(|_| D::Error::custom(concat!($label, "_encoding_invalid")))?;
                Self::new(<$float>::from_bits(bits)).map_err(D::Error::custom)
            }
        }
    };
}

finite_float!(FiniteF32, f32, u32, 8, "finite_f32");
finite_float!(FiniteF64, f64, u64, 16, "finite_f64");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_wire_roundtrip_preserves_neighbor_sensitive_values() {
        let value = FiniteF64::new(0.9376332445278925).unwrap();
        let encoded = serde_json::to_vec(&value).unwrap();
        let reopened: FiniteF64 = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(value.bits(), reopened.bits());
    }

    #[test]
    fn rejects_nonfinite_and_noncanonical_wire_values() {
        assert!(FiniteF32::new(f32::NAN).is_err());
        assert!(serde_json::from_str::<FiniteF32>("\"7F800000\"").is_err());
        assert!(serde_json::from_str::<FiniteF64>("\"7ff0000000000000\"").is_err());
    }
}
