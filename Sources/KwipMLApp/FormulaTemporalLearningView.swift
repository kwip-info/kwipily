import SwiftUI
import KwipMLCore

struct FormulaTemporalLearningView: View {
    @ObservedObject var store: WorkspaceStore
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            LessonTextPanel(title: "Date/time · editable practice", text: "Turn local appointment text into real instants, compare elapsed time and extract useful features. This practice is unscored.")
            LessonTextPanel(title: "Predict before running", text: "**1. Parse:** The first two appointments both read 01:30 in New York when the autumn clock repeats. Earliest selects 05:30 UTC; Latest selects 06:30 UTC. Their Difference is 3,600 seconds.\n\n**2. Format:** Convert an instant to UTC, then Format it. Conversion preserves the instant; formatting creates Text.\n\n**3. Missing:** The spring 02:30 appointment never happened on that clock. Stop reports the row; Missing preserves the row with a missing value.\n\n**4. Precision:** One microsecond after the epoch is exactly 1 in canonical microsecond storage. A seventh fractional digit must fail rather than round.\n\n**5. Extract:** Hour in New York and Hour in UTC can differ for the same instant. Choose the timezone explicitly before creating a model feature.")
            Button("Open date/time practice") { _ = store.loadComponentWorkflowExample(FormulaTemporalPracticeExample.exampleID, device: "cpu") }
                .buttonStyle(FlatButtonStyle()).disabled(store.isBusy)
                .accessibilityIdentifier("learning.formula.temporal.open-example")
        }
    }
}
