use serde_json::json;
use std::path::Path;
use trainaya::config::MergeMode;
use trainaya::dataset::DecisionDataset;
use trainaya::families::traits::DecisionSample;
use trainaya::families::FamilyRegistry;

#[test]
fn test_family_registry_lookup_and_detection() {
    let registry = FamilyRegistry::default();

    // Direct lookup
    assert!(registry.get("winnow-v1").is_some());
    assert!(registry.get("kev-pointer-v1").is_some());
    assert!(registry.get("nli-pairs-v1").is_some());
    assert!(registry.get("laya-markers-v1").is_some());
    assert!(registry.get("llm-logits-v1").is_some());

    // Auto-detection by name pattern
    let winnow = registry.detect_or_resolve("winnow:e4b", None).unwrap();
    assert_eq!(winnow.identifier(), "winnow-v1");
    assert_eq!(winnow.default_merge_mode(), MergeMode::Merge);

    let kev = registry.detect_or_resolve("kev:0.8b", None).unwrap();
    assert_eq!(kev.identifier(), "kev-pointer-v1");
    assert_eq!(kev.default_merge_mode(), MergeMode::NoMerge);

    let nli = registry.detect_or_resolve("nli:deberta-v3", None).unwrap();
    assert_eq!(nli.identifier(), "nli-pairs-v1");

    let laya = registry.detect_or_resolve("laya:custom", None).unwrap();
    assert_eq!(laya.identifier(), "laya-markers-v1");

    // Explicit override takes precedence
    let forced = registry
        .detect_or_resolve("some_random_model", Some("kev-pointer-v1"))
        .unwrap();
    assert_eq!(forced.identifier(), "kev-pointer-v1");
}

#[test]
fn test_prompt_builders_formatting() {
    let registry = FamilyRegistry::default();
    let sample = DecisionSample {
        state: json!("High memory watermark exceeded on node worker-03"),
        question: "How should this alert be routed?".to_string(),
        options: vec![
            "page_sre_immediate".to_string(),
            "silent_auto_ack".to_string(),
        ],
        target_index: 1,
    };

    // 1. Winnow prompt builder
    let winnow_family = registry.get("winnow-v1").unwrap();
    let winnow_prompt = winnow_family
        .create_prompt_builder()
        .format_sample(&sample)
        .unwrap();
    assert!(winnow_prompt.text.contains("A: page_sre_immediate"));
    assert!(winnow_prompt.text.contains("B: silent_auto_ack"));
    assert!(winnow_prompt.text.contains("Answer:\n"));
    assert_eq!(winnow_prompt.target_index, 1);

    // 2. Kev prompt builder (delimiters)
    let kev_family = registry.get("kev-pointer-v1").unwrap();
    let kev_prompt = kev_family
        .create_prompt_builder()
        .format_sample(&sample)
        .unwrap();
    assert!(kev_prompt.text.contains("<|fim_prefix|>High memory"));
    assert!(kev_prompt.text.contains("<|fim_middle|>How should"));
    assert!(kev_prompt.text.contains("<|box_start|>page_sre_immediate<|box_end|>"));
    assert!(kev_prompt.text.contains("<|box_start|>silent_auto_ack<|box_end|>"));
    assert!(kev_prompt.text.ends_with("<|fim_suffix|>"));

    // 3. NLI prompt builder (pairwise [SEP])
    let nli_family = registry.get("nli-pairs-v1").unwrap();
    let nli_prompt = nli_family
        .create_prompt_builder()
        .format_sample(&sample)
        .unwrap();
    assert!(nli_prompt.text.contains("[SEP]"));
    assert!(nli_prompt.text.contains("silent_auto_ack"));
}

#[test]
fn test_decision_dataset_schema_loading() {
    let schema_file = Path::new("data/questions_schema.json");
    let data_file = Path::new("data/train_specific_decision.jsonl");

    if schema_file.exists() && data_file.exists() {
        let samples = DecisionDataset::load_samples(data_file, Some(schema_file)).unwrap();
        assert!(!samples.is_empty());

        // Each input record creates samples for the schema questions (routing and is_breach)
        let first = &samples[0];
        assert!(!first.question.is_empty());
        assert!(!first.options.is_empty());
        println!("Sample question: {}", first.question);
        println!("Sample options: {:?}", first.options);
    }
}
