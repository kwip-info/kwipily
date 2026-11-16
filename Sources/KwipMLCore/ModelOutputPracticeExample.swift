import Foundation

/// A runnable practice with the real Tensor → Definition → Model definition boundary.
/// This is guided practice; opening or running it does not award assessed mastery.
public enum ModelOutputPracticeExample {
    public static let exampleID = "model-output"
    public static let observationsResource = "model-output-observations"
    public static let address = WorkflowNodeAddress(domain: .model, localID: "output")
    public static let title = "Check a model’s output without changing it"
    public static let objective = "Confirm that the last tensor connection matches the model’s configured output expectation."
    public static let firstCanvasAction = "1. Click [Signal observations](kwip-node://data/observations) to bring it into view. Select the block to inspect its source in Configuration.\n2. Trace the upper connections to see how the data reaches Train. Then follow the lower model path from [Two scaled features](kwip-node://model/input) to [Model Output](kwip-node://model/output). Use the colored names above to find each block. No configuration changes or run are needed for this step."
    public static let understanding = "If the model returns a tensor with shape [4, 2], how many observations did it process, and how many scores did it produce for each one? Why would changing the batch to one observation give [1, 2] rather than [1, 1]? If we configure the model to expect three scores, will that create a third score?"
    public static let understandingAnswer = "The 4 counts observations processed together. The 2 counts scores per observation: one for group 0 and one for group 1. A batch of one still needs both scores, so its shape is [1, 2]. Changing the expected size only checks what arrives; it does not add a score. To produce three scores, you would need to change the model itself."
    public static let instructions = """
    **What Model Output does**
    This canvas is a small classification example: a model learns to put a row into one of two groups. You might use classification to sort messages or flag unusual measurements. Here we use made-up numbers so you can focus on how the workflow works, without needing a real dataset.
    [Signal observations](kwip-node://data/observations) reads 12 rows. Each row has two measurements, signal and context, plus a label telling us its known group: 0 or 1.
    - [Columns](kwip-node://data/cast): chooses these fields.
    - [Partition](kwip-node://data/split): reserves different rows for learning, validation and a final test.
    - [Preprocess](kwip-node://data/fit): scales the measurements using the training rows.
    - [Inspect prepared training observations](kwip-node://data/training) and [Inspect prepared validation observations](kwip-node://data/validation): show the prepared tables.
    The model is made of these blocks:
    - [Two scaled features](kwip-node://model/input): brings the two prepared measurements into the model.
    - [Parallel feature paths](kwip-node://model/features): processes them along two paths and combines what they learn.
    - [Class scores](kwip-node://model/classifier): produces two numbers for each row, a score for group 0 and a score for group 1.
    - [Fit and evaluate classifier](kwip-node://control/train): uses the known labels to adjust the model’s weights.
    A higher score is the model’s preference; these scores are not automatically probabilities, and a successful run does not prove it predicts well on new data.
    A tensor is an array of numbers. In this example, think of the output as a table: one row per observation, with two score columns. Shape [4, 2] means 4 observations processed together and 2 scores for each observation. Processing rows together is called a batch. With one observation, the shape is [1, 2]: the number of rows changes, but each row still has two scores. The per-observation, or sample, shape is [2].
    [Model Output](kwip-node://model/output) is the name of this canvas’s Model definition block. It receives the final scores and tells Train which model to use. Its optional output check confirms that the last tensor connection matches the model’s configured expectation. It checks the numbers’ format and dimensions without changing their values.

    **Follow the result**
    1. Find the connection from [Class scores](kwip-node://model/classifier) to [Model Output](kwip-node://model/output). This connection carries the two scores for each observation. [Model Output](kwip-node://model/output)’s Model definition connection leads to [Fit and evaluate classifier](kwip-node://control/train), the Train block.
    2. Select [Model Output](kwip-node://model/output) and open Configuration → Expected output. Leave Model inputs unchanged for this exercise. Keep Check expected output enabled. Expected value type should be float32, a format for decimal numbers. Expected sample rank should be Exact, with one Sample dimension set to Exact and size 2. Here rank means the number of dimensions for one observation; one dimension of size 2 means two scores. The batch size is not part of this setting. Apply any pending changes.
    3. Run Flow. When it finishes, select the [Class scores](kwip-node://model/classifier) output connection to inspect its tensor in Results. Read the shape as observations × scores per observation. The viewer shows recorded values from this run; it cannot show values that were not recorded. A completed check means the output has the expected format and dimensions, not that every predicted group is correct.

    **Try a deliberate mismatch**
    Select [Model Output](kwip-node://model/output). Change its expected dimension size from 2 to 3 and Apply, then run again. The model still produces two scores per observation, so the output check reports a mismatch. Read the finding in Diagnostics, then restore size 2, Apply and run again. Changing an expectation does not create another score or change a prediction; it helps you catch a model that produces a different output than you intended.

    **Practice independently**
    Explain what the two score columns represent, and why each observation needs both even when the batch size changes. Then try Any for the sample dimension: the check will allow that dimension’s size to vary. Any sample rank also allows the number of sample dimensions to vary, while still checking the selected value type. Turning off Check expected output skips this optional check; it does not change the model’s calculation. Restore Exact with one dimension of size 2 when you finish.

    This is an unscored practice. Use it to learn how to read and check a model’s output. Opening or running it does not award assessment completion or establish that a model is accurate.
    """
    public static func resourceURL() throws -> URL {
        guard let url = Bundle.module.url(forResource: observationsResource, withExtension: "csv") else { throw GraphConfigurationError("Model Output practice observations are missing.") }
        return url
    }
    public static func document(device: String = "auto") throws -> WorkflowDocument {
        var document = try WorkflowExamples.preparedParallelClassifier(device: device)
        document.name = title
        guard let index = document.nodes.firstIndex(where: { $0.address == address }) else { throw GraphConfigurationError("Model Output practice boundary is missing.") }
        document.nodes[index].name = "Model Output"
        var config = try ModelOutputConfiguration([:])
        config.checkExpectedOutput = true; config.expectedDType = "float32"
        config.expectedSampleRank = "exact"; config.expectedSampleDimensions = [2]
        document.nodes[index].payload = .modelPrimitive(type: "output", configurationVersion: 2, config: config.values)
        try WorkflowCodec.validate(document)
        return document
    }
}
