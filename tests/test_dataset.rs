use std::path::Path;
use trainaya::dataset::Dataset;

#[test]
fn test_load_jsonl_dataset() {
    let jsonl_path = Path::new("examples/sample_classification.jsonl");
    assert!(jsonl_path.exists());

    let samples = Dataset::load_raw_samples(jsonl_path).expect("Failed to load JSONL");
    assert_eq!(samples.len(), 12);
    assert_eq!(samples[0].label, "positive");
    assert_eq!(samples[1].label, "negative");
}

#[test]
fn test_load_nli_pairs() {
    let nli_path = Path::new("examples/sample_nli.jsonl");
    assert!(nli_path.exists());

    let samples = Dataset::load_raw_samples(nli_path).expect("Failed to load NLI JSONL");
    assert_eq!(samples.len(), 6);
    assert!(samples[0].text_b.is_some());
    assert_eq!(samples[0].label, "entailment");
}

#[test]
fn test_load_csv_dataset() {
    let csv_path = Path::new("examples/sample_classification.csv");
    assert!(csv_path.exists());

    let samples = Dataset::load_raw_samples(csv_path).expect("Failed to load CSV");
    assert_eq!(samples.len(), 4);
    assert_eq!(samples[0].label, "positive");
}

#[test]
fn test_split_dataset() {
    let jsonl_path = Path::new("examples/sample_classification.jsonl");
    let samples = Dataset::load_raw_samples(jsonl_path).unwrap();

    let (train, eval) = Dataset::split_samples(samples, 0.25, 42);
    assert_eq!(train.len(), 9);
    assert_eq!(eval.len(), 3);
}
