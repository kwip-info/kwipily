import Foundation

@MainActor enum DataExampleLessonCopy {
    static let joinedIntroduction="""
    A measurement and its context may arrive in different files. Matching row 1 to row 1 is unsafe when those files use different orders. A shared observation_id lets Join Tables match the right records.

    This Example initially opens a lesson canvas. Use Open joined example in Exercise choices & reference to load the supplied graph. Save existing work before replacing it.

    The graph contains:
    - A signal table and Context by observation, whose rows are deliberately reordered.
    - Match observation identities, a one-to-one inner join on observation_id.
    - Columns, Partition and Preprocess to prepare numeric features using Training statistics.
    - Model input, learnable layers and Model Output defining the classifier.
    - Train, which learns from Training and checks Validation.

    An inner join retains matching keys from both sources. One-to-one requires each key to occur at most once on either side; it prevents repeated keys from silently multiplying observations.
    """
    static let aggregatedIntroduction="""
    Repeated measurements of one person are related. If those measurements are split independently, the model could see the same person during both training and evaluation.

    This Example initially opens a lesson canvas. Use Open aggregation example in Exercise choices & reference to load the supplied graph. Save existing work first.

    Follow the table:
    - Three measurements per subject reads 192 synthetic measurements for 64 subjects.
    - Columns converts the numeric fields.
    - One observation per subject uses Summarize to calculate mean signal/context, label minimum/maximum and measurement count.
    - Partition separates the resulting subjects before Preprocess learns from Training.
    - The classifier and Train learn a two-class prediction from the prepared features.

    Here each summary represents one subject, not one original measurement. This is a deliberate choice about what an observation means.
    """
    static func steps(_ id:String)->[LessonContentStep]? {
        switch id {
        case "join": return [
            LessonContent.step("goal","Match observations by identity",joinedIntroduction),
            LessonContent.step("build","Inspect matching keys","1. Open the supplied joined example below.\n2. Select Match observation identities.\n3. In Matching keys, confirm observation_id is paired with observation_id.\n4. Inspect the one-to-one key relationship and inner join choice.\n\nMatching uses key values, not row positions. Both sides need compatible key types; Columns makes a deliberate conversion when required.",kind:.canvasAction),
            LessonContent.step("join.columns","Choose the output fields","Inspect Output columns before editing. The two original observation_id fields establish matching even if an output copy is excluded. Keep the fields required by the downstream Columns and Train mapping.\n\nTry renaming or reordering a non-key output, Apply and inspect any downstream finding. Restore the original names before training. Changing a name does not automatically rewrite a later expression or mapping.",kind:.canvasAction),
            LessonContent.step("join.run","Run and trace the prepared table","1. Run Flow with the supplied settings.\n2. Inspect the joined table and compare a few observation IDs across both sources.\n3. Follow the rows through Partition and Preprocess.\n4. Select Train's predictions output to inspect captured Validation results.\n\nTraining results depend on the actual run. A successful join does not establish that the model predicts accurately.",kind:.canvasAction),
            LessonContent.step("inspect","Explain the matching rule","Why does reordering Context by observation leave the matches correct? What would repeated keys on both sides do without a relationship check? Why do repeated measurements of one subject need a different evaluation split?\n\nThis is an editable, unscored example. Describe the captured evidence rather than treating opening or running as an assessment pass.",kind:.question)
        ]
        case "aggregate": return [
            LessonContent.step("goal","Choose one observation per subject",aggregatedIntroduction),
            LessonContent.step("build","Inspect grouping and summaries","1. Open the supplied aggregation example below.\n2. Select One observation per subject.\n3. Inspect Group by subject_id and each Aggregations row.\n\nMean produces the average signal/context for a subject. Count rows records how many measurements contributed. Minimum and maximum label are separate checks on whether that subject's recorded answers agree.",kind:.canvasAction),
            LessonContent.step("aggregate.run","Compare measurements with subjects","1. Run Flow with the supplied settings.\n2. Inspect the source's 192 measurements and Summarize's 64 output rows.\n3. Check measurement_count is three for each subject.\n4. Compare label with label_check.\n\nThe supplied labels agree. Minimum and maximum expose disagreements in other data; Summarize does not automatically reject or repair them.",kind:.canvasAction),
            LessonContent.step("aggregate.train","Follow the evaluation boundary","Follow the summaries into Partition, then the Training and Validation paths through Preprocess to Train. Each subject summary belongs to one partition. Preparation statistics come from Training only.\n\nInspect Train's captured predictions for the actual Validation result. Compare the evaluated population with the subject summaries; a measurement count and an evaluated subject count describe different things.",kind:.canvasAction),
            LessonContent.step("inspect","Explain your observation unit","Why summarize before splitting these repeated measurements? When might averaging hide a useful change over time? What should you do if a subject's minimum and maximum labels differ?\n\nThis editable example is unscored. Its labels and three-measurement groups describe this supplied dataset, not a promise about other sources.",kind:.question)
        ]
        default:return nil
        }
    }
}
