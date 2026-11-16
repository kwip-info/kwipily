/// Beginner presentation for the existing classifier course. This does not alter
/// graph placement, saved course identity, assessment rules or execution settings.
enum ClassifierLessonGuide {
    static func explanation(for nodeID: String, fallback: String,version:Int=5) -> String {
        if version>=5,let current=GuidedLessonCopy.stages[nodeID] {return current.text}
        if version<5 && nodeID=="mapping" {return "Select the saved Training data mapping. Keep signal then context as inputs and label as target; its separate Training and Validation tables retain their population roles. This saved course uses its original mapping boundary."}
        if version<5 && nodeID=="train" {return "Train receives the model and saved mapped populations. Its Configuration owns run settings. Run and inspect the captured loss, metrics and validation predictions; do not tune on the Test population."}
        return explanations[nodeID] ?? fallback
    }

    static func title(for nodeID:String,fallback:String,version:Int=5)->String {
        version>=5 ? GuidedLessonCopy.stages[nodeID]?.title ?? fallback:fallback
    }

    private static let explanations: [String: String] = [
        "observations": """
        **What this does**
        Start with a table. Each row is one observation: one example the model can learn from. Features are the input columns the model uses. The target is the answer you want it to predict. Here, signal and context are features; label is the target.

        **Try it**
        Select Table Input on the canvas and look at its Configuration. As you place the next stages, follow its output connector. After the complete flow runs, select the CSV source and open Results → Output to see the captured rows and columns. For a specific port, use Results → Ports & connections → Inspect.
        """,
        "cast": """
        **What this does**
        Casting tells the app what kind of values each column holds. Floating-point numbers can include decimals, so they suit measurements. Integers are whole numbers; here they identify the two answer categories. A column's number type is separate from its role as a feature or target.

        **Try it**
        Select Columns and check its grid: signal and context use float64, while label uses int64. These are the app's type names for decimal and whole-number values. Keep those choices for this example. After running, inspect the output table's column types.
        """,
        "split": """
        **What this does**
        Training rows teach the model. Validation rows are reserved for checking how it handles examples it did not train on. Partition reserves 25% for validation and 25% for final Test, leaving half for training. A seed is a number used to repeat a random choice with the same data and settings.

        **Try it**
        Select Partition and find both held-out percentages and the seed in Configuration. Keep 25% for Validation, 25% for Test and seed 42 for your first run. Follow the separate training and validation connectors. Validation helps you compare changes; a final test set should remain separate from those development decisions.
        """,
        "fit": """
        **What this does**
        A scaler brings feature values onto a more comparable scale. It learns each feature's mean (average) and spread from training rows. Fitting means learning those rules from data. The saved rules are called fitted state.

        **Try it**
        Select Preprocess. Check that signal and context use Standard scaling and follow its Training, Validation and Test inputs back to Partition. The statistics are learned from Training only and reused for all three outputs. Do not use validation rows to learn the scaling rules: that would let checking data influence training.
        """,
        "training": """
        **What this does**
        Preprocess applies its saved rules to the training features. It subtracts the training average and divides by the learned spread. The target column stays the answer; it is not a feature to scale.

        **Try it**
        Select the training View and trace its input to Preprocess’s Training output. Once the full flow has run, inspect its output table. Compare the feature values with the source; the rows still represent the same observations, but their feature values use the new scale.
        """,
        "validation": """
        **What this does**
        Validation needs the same scaling rules as training. Imagine measuring one group in meters and another in centimeters: comparing them directly would be misleading. Sharing the training scaler keeps both groups in the same system.

        **Try it**
        Select the validation View. Follow its input to Preprocess’s Validation output; it uses the same training-fitted statistics. After running, inspect the transformed validation rows. Do not fit a second scaler on them.
        """,
        "input": """
        **What this does**
        Input declares what enters the model. A tensor is a table or a larger array of numbers. This model receives two feature values per observation, in a fixed order. A batch is a small group of rows processed together.

        **Try it**
        Select Input and check that Features is 2. Follow its outgoing connectors into the two model branches. The number of rows can vary between batches; each row must still contain the same two feature columns.
        """,
        "branch-a": """
        **What this does**
        Linear combines input values using learned weights: numbers that control how strongly each input contributes. A bias adds a learned offset. For example, two inputs 2 and 3 with weights 1 and 2 and an offset of 1 produce 2 + 6 + 1 = 9. Training adjusts these numbers.

        **Try it**
        Select the first Linear component. Check its Configuration: 2 input features become 8 output features. These eight values are new combinations of the two original measurements. Leave the widths unchanged for the first run.
        """,
        "activation-a": """
        **What this does**
        ReLU keeps positive values and changes negative values to zero. For example, -3 becomes 0 and 2 stays 2. This bends the model's response, helping it learn patterns that weighted sums alone cannot represent. An activation is a rule that changes values this way.

        **Try it**
        Select ReLU and follow its input from the first Linear component. It keeps the same number of features. After running, inspect its captured output if available and look for zeros where negative values were removed.
        """,
        "branch-b": """
        **What this does**
        The second branch reads the same two input features but learns its own weights. Two paths sharing input data do not automatically share learned numbers. They can develop different combinations of the original measurements.

        **Try it**
        Select the second Linear component and check its 2-to-8 feature settings. Trace its input to the same Input component as the first branch. Compare the two paths after running; the structure matches, but their learned responses can differ.
        """,
        "activation-b": """
        **What this does**
        Tanh smoothly squeezes values between -1 and 1. Zero stays zero; large positive values approach 1 and large negative values approach -1. Unlike ReLU, it keeps a negative response. A bounded output does not guarantee that training will go well.

        **Try it**
        Select Tanh and follow its input from the second Linear component. After running, inspect the captured values if available and compare their range with the ReLU branch. Both paths still produce eight features per observation.
        """,
        "merge": """
        **What this does**
        Concatenate means join side by side. The two branches each contribute eight features, so joining them gives 16 per observation. Add would instead sum corresponding values and keep eight features; these are different operations.

        **Try it**
        Select Concatenate and follow its two input connectors to ReLU and Tanh. Check that both branches arrive in the Connected inputs grid. Join axis −1 selects the final feature direction; Planned shape is declared, and Output shows captured values after a run. Its output keeps both sets of features for the next Linear component rather than adding them together.
        """,
        "classifier": """
        **What this does**
        The final Linear component combines the 16 joined features into two class scores, one for each possible answer. Raw scores are also called logits. They are not probabilities: they do not need to lie between 0 and 1 or add up to 1.

        **Try it**
        Select this Linear component and check 16 input features and 2 output features. Follow its output to Model definition. After running, inspect the captured scores if available; a higher score is the model's preference, not proof that its answer is correct.
        """,
        "output": """
        **What this does**
        Model definition marks the model's answer values. Here, every observation produces two raw class scores. The training loss compares these scores with the known target and handles the probability calculation it needs.

        **Try it**
        Select Model definition and follow its input to the final Linear component. Keep the two raw scores as the output. Do not add Softmax here: Softmax turns scores into probabilities, but this training setup expects the raw scores.
        """,
        "model": """
        **What this does**
        Model definition identifies the connected model architecture: the arrangement of components that turns features into scores. Its connection carries a model description, not rows from the data table. The Train component needs both the model and its data.

        **Try it**
        Select Model definition: its tensor input comes from the final Linear layer, and its Model definition output connects directly to Train. Configure input features and entry ports here; no separate pass-through block is needed.
        """,
        "mapping": """
        **What this does**
        Mapping tells the model which columns are inputs and which column is the answer. Column order matters: a model trained on signal then context must receive them in that order when it predicts.

        **Try it**
        Select Train. In Configuration → Data mapping, keep signal then context as the two feature columns and label as the target. Follow the Training and Validation inputs to Preprocess’s corresponding table outputs. Do not include label among the input features; that would give the model the answer.
        """,
        "setup": """
        **What this does**
        Train’s Run settings control how training happens. An epoch is one pass through the training rows. A batch is a small group of rows used for one update. The learning rate controls the size of the adjustments to learned weights. A seed helps repeat the model's starting random choices; Partition has its own seed.

        **Try it**
        Select Train and find Device, Epochs and Seed in Run settings, with batch and optimizer controls in Configuration. Begin with the example's 3 epochs, batch size 32, seed 42 and learning rate 0.03. Device selects the supported processor for the run. Later, change one setting at a time and compare recorded results in History.
        """,
        "train": """
        **What this does**
        Train accepts the Model definition and the prepared Training and Validation tables directly. In Configuration → Data mapping, choose signal then context as inputs and label as target. Column order must match the model’s input settings. Train brings this mapping and run settings together. Loss measures how wrong the predictions are under the chosen training rule; lower is generally better for the same task. Accuracy is the fraction of answers predicted correctly. Validation results show performance on rows kept out of training.

        **Try it**
        Check that Train has its Model and Data inputs connected. In Configuration → Run settings, choose device, epochs, batch size and seed. Optimizer settings own their learning rate. Choose Run Flow. Clear the canvas selection to watch Results → Activity. Select a data component and open Results → Output to inspect its captured table, and open History to compare training and validation results. A successful run is evidence that this flow works; this guided practice does not assess mastery or guarantee performance on other data.
        """
    ]
}
