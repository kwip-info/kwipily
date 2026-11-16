import SwiftUI

struct PartitionInformationView:View {
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            Label("Three independent populations",systemImage:"arrow.triangle.branch").fontWeight(.semibold).foregroundStyle(MLTheme.teal)
            Text("Training fits the model and preparation statistics. Validation guides choices. Keep Test untouched until final evaluation. All three outputs must contain observations.")
            Label("Random or whole entities",systemImage:"shuffle").fontWeight(.semibold)
            Text("Random ranks stable observation identities using the seed. Optional stratification allocates within each scalar value. By entity keeps every row sharing the ordered keys together. Percentages allocate entities, so row proportions may differ. Integer allocation uses largest remainders; small strata that cannot fill all three outputs stop with a diagnostic.")
            Label("Chronological boundaries",systemImage:"clock").fontWeight(.semibold)
            Text("Training includes its cutoff. Validation starts strictly after the training cutoff plus embargo and includes its own cutoff. Test starts strictly after the validation cutoff plus embargo. Exclusions contains embargo-gap rows for inspection in Results, without an outgoing connection. Retained time outputs are sorted by instant and observation identity.")
            Text("Use explicit-offset ISO timestamps or integer epoch milliseconds. Entity keys and stratification must be nonmissing scalar values. Require disjoint entities rejects an entity appearing in multiple retained time partitions. Resolve missing keys or invalid times upstream; Partition never silently drops them.")
            Label("Apply & inspect",systemImage:"checkmark.circle").fontWeight(.semibold)
            Text("Apply saves the active strategy as one undoable edit. Switching strategies preserves drafts until you leave this inspector; unapplied changes block Run. Results exposes each complete captured table, including Exclusions. Global Working data can restrict a sample run; View, training and file outputs request complete upstream data. Existing Split and legacy partition nodes keep their original behavior.")
        }
    }
}
