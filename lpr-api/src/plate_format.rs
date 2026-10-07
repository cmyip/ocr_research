//! Format rules that settle characters the OCR cannot tell apart by shape.
//!
//! Malaysian plates draw `1` as a bare stroke and `0` as an oval, the same shapes as `I` and `O`, so
//! only the position on the plate says which one it is: letters first, then a number of one to four
//! digits, then at most one suffix letter (`VGG 811`, `UITM 1776`, `WB 110 J`).

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlateFormat {
    /// Report the OCR's characters as they are.
    #[default]
    None,
    /// Malaysia: letters, then 1-4 digits, then at most one letter. Only I/1 and O/0 are ever changed.
    My,
}

/// Special series whose prefix has digits in it; they are left alone ahead of the number.
const MY_DIGIT_PREFIXES: [&str; 5] = ["1M4U", "G1M", "T1M", "A1M", "K1M"];

pub fn apply(format: PlateFormat, text: &str) -> String {
    match format {
        PlateFormat::None => text.to_string(),
        PlateFormat::My => malaysia(text),
    }
}

fn as_letter(c: u8) -> Option<u8> {
    match c {
        b'1' => Some(b'I'),
        b'0' => Some(b'O'),
        c if c.is_ascii_uppercase() => Some(c),
        _ => None,
    }
}

fn as_digit(c: u8) -> Option<u8> {
    match c {
        b'I' => Some(b'1'),
        b'O' => Some(b'0'),
        c if c.is_ascii_digit() => Some(c),
        _ => None,
    }
}

/// The reading of `text` as letters + number + optional suffix that changes the fewest characters.
/// A read that already fits is returned unchanged; one that cannot fit is returned unchanged too.
fn malaysia(text: &str) -> String {
    let fixed = MY_DIGIT_PREFIXES.iter().find(|p| text.starts_with(**p)).map_or(0, |p| p.len());
    let s = text.as_bytes();
    let n = s.len();
    if !s.iter().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit()) {
        return text.to_string();
    }
    let mut best: Option<(usize, usize, Vec<u8>)> = None; // (changes, -digits, reading)
    for suffix in 0..=1usize {
        for digits in 1..=4usize {
            if n < fixed.max(1) + digits + suffix {
                continue;
            }
            let letters = n - digits - suffix;
            let mut out = s[..fixed].to_vec();
            out.extend(s[fixed..letters].iter().filter_map(|&c| as_letter(c)));
            out.extend(s[letters..letters + digits].iter().filter_map(|&c| as_digit(c)));
            // a suffix is a letter of the ordinary series, which has no I or O
            out.extend(s[letters + digits..].iter().filter(|c| c.is_ascii_uppercase() && !matches!(**c, b'I' | b'O')));
            if out.len() != n || out[letters] == b'0' || !s[..letters].iter().any(u8::is_ascii_uppercase) {
                continue; // no reading in that position, a number starting with 0, or no letter the OCR was sure of
            }
            let changes = out.iter().zip(s).filter(|(a, b)| a != b).count();
            let key = (changes, 4 - digits);
            if best.as_ref().is_none_or(|b| key < (b.0, b.1)) {
                best = Some((key.0, key.1, out));
            }
        }
    }
    best.map_or_else(|| text.to_string(), |b| String::from_utf8(b.2).unwrap_or_else(|_| text.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn my(s: &str) -> String {
        apply(PlateFormat::My, s)
    }

    #[test]
    fn a_stroke_in_the_number_is_a_one_and_in_the_letters_an_i() {
        assert_eq!(my("VGG8I"), "VGG81");
        assert_eq!(my("VGG8II"), "VGG811");
        assert_eq!(my("PKG2III"), "PKG2111");
        assert_eq!(my("WB1I0J"), "WB110J");
        assert_eq!(my("U1TM1776"), "UITM1776");
        assert_eq!(my("PR0T0N12"), "PROTON12");
        assert_eq!(my("BRL4I04"), "BRL4104");
        assert_eq!(my("W1O"), "W10");
    }

    #[test]
    fn reads_that_already_fit_are_left_alone() {
        for s in ["UITM1776", "VGG811", "WB110J", "PUTRAJAYA541", "EV232", "BRL4104", "IIUM1", "IO10", "G1M1234", "1M4U88"] {
            assert_eq!(my(s), s);
        }
    }

    #[test]
    fn reads_that_cannot_fit_are_left_alone() {
        for s in ["", "ABC", "12345", "1234", "AB12CD", "W1234AB"] {
            assert_eq!(my(s), s);
        }
        assert_eq!(apply(PlateFormat::None, "VGG8I"), "VGG8I");
    }
}
