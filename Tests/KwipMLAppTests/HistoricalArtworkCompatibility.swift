import XCTest

/// Explicit reviewed consolidation aliases for frozen artwork-history receipts.
/// No original digest changes; unrelated mappings remain visible to the comparison.
enum HistoricalArtworkCompatibility {
    enum Failure: Error { case unapprovedTableInputAlias }
    static func mapping(_ current: [String: String]) throws -> [String: String] {
        guard current["kwip.data.sourceCSV"] == "table-input-v1" else {
            throw Failure.unapprovedTableInputAlias
        }
        var historical = current
        historical["kwip.data.sourceCSV"] = "csv-table-v2"
        let approved: [String:(String,String)] = [
            "workflow.logisticRegression":("logistic-regression-v2","prebuilt-models-v1"),
            "workflow.kerasClassifier":("keras-classifier-v2","prebuilt-models-v1"),
            "input":("input-v2","model-input-v3"),
            "index_input":("index_input-v2","model-input-v3"),
            "batch_norm1d":("batch_norm1d-v2","normalization-v3"),
            "batch_norm2d":("batch_norm2d-v2","normalization-v3"),
            "bilinear":("bilinear-v2","bilinear-v3"),
            "conv1d":("conv1d-v2","conv2d-v3"),
            "conv2d":("conv2d-v2","conv2d-v3"),
            "conv3d":("conv3d-v2","conv2d-v3"),
            "conv_transpose1d":("conv_transpose1d-v2","conv2d-v3"),
            "conv_transpose2d":("conv_transpose2d-v2","conv2d-v3"),
            "conv_transpose3d":("conv_transpose3d-v2","conv2d-v3"),
            "depthwise_conv2d":("depthwise_conv2d-v2","conv2d-v3"),
            "dropout":("dropout-v2","dropout-v3"),
            "group_norm":("group_norm-v2","normalization-v3"),
            "gru":("gru-v2","rnn-v2"),
            "gru_cell":("gru_cell-v2","rnn_cell-v2"),
            "instance_norm1d":("instance_norm1d-v2","normalization-v3"),
            "instance_norm2d":("instance_norm2d-v2","normalization-v3"),
            "layer_norm":("layer_norm-v2","normalization-v3"),
            "lazy_linear":("lazy_linear-v2","linear-v2"),
            "lstm":("lstm-v2","rnn-v2"),
            "lstm_cell":("lstm_cell-v2","rnn_cell-v2"),
            "output":("output-v2","model-boundary-v3"),
            "separable_conv2d":("separable-conv2d-v2","separable-conv2d-v3"),
            "workflow.modelDefinition":("model-definition-v2","model-boundary-v3")
        ]
        for (key,pair) in approved where current[key] != nil {
            guard current[key] == pair.1 else {throw Failure.unapprovedTableInputAlias}
            historical[key] = pair.0
        }
        let aliases: Set<String> = ["kwip.activation","kwip.pooling","kwip.convolution","kwip.normalization","kwip.linear","kwip.recurrent","kwip.recurrentCell","kwip.reshape","kwip.tensor.axes","kwip.tensor.parts","parallel_features","workflow.saveModel"]
        for alias in aliases {historical.removeValue(forKey:alias)}
        // These additive primitives are outside the frozen pre-text artwork cohort.
        for (key,asset) in ["rms_norm":"normalization-v3","rotary_position":"rotary_position-v2",
            "mixture_experts":"mixture-experts-v1","workflow.gameSession":"game-session-v1",
            "kwip.data.gameRules":"data-game-rules-v1","kwip.data.decide":"data-decide-v1",
            "kwip.data.classifyAudio":"data-classify-audio-v1","kwip.data.transcribeAudio":"data-transcribe-audio-v1",
            "kwip.data.extractText":"data-extract-text-v1","kwip.data.prepareMedia":"data-prepare-media-v1"] where current[key] != nil {
            guard current[key]==asset else {throw Failure.unapprovedTableInputAlias}
            historical.removeValue(forKey:key)
        }
        // Load model intentionally uses the reviewed native tray/frame. The PNG
        // remains a historical reference, not a live generated-artwork mapping.
        if current["workflow.train"] != nil {
            guard current["workflow.frozenModel"] == nil else {throw Failure.unapprovedTableInputAlias}
            historical["workflow.frozenModel"] = "frozen-model-v2"
        }
        return historical
    }
}

final class HistoricalArtworkCompatibilityTests: XCTestCase {
    func testOnlyTheApprovedAliasChangesAndCallerIsUntouched() throws {
        let current = ["kwip.data.sourceCSV":"table-input-v1", "linear":"altered-value-still-visible"]
        let normalized = try HistoricalArtworkCompatibility.mapping(current)
        XCTAssertEqual(normalized, ["kwip.data.sourceCSV":"csv-table-v2", "linear":"altered-value-still-visible"])
        XCTAssertEqual(current["kwip.data.sourceCSV"], "table-input-v1")
    }
    func testMissingHistoricalAndUnapprovedAliasesAreRefused() {
        for mapping in [[:], ["kwip.data.sourceCSV":"csv-table-v2"], ["kwip.data.sourceCSV":"foreign"]] {
            XCTAssertThrowsError(try HistoricalArtworkCompatibility.mapping(mapping))
        }
    }
}
