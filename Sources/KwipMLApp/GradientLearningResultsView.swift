import SwiftUI
import KwipMLCore

/// Public measured comparisons only. The verified Core projection has already withheld private assessments.
struct GradientLearningResultsPresentation {
    struct Table {
        let key: String
        let title: String
        let meaning: String
        let measured: LearningTensor
        let reference: LearningTensor
    }
    struct Fixture {
        let id: String
        let tables: [Table]
    }
    let fixtures: [Fixture]
    static let introduction = "These tables show measured comparisons with an independent reference. A difference is measured minus reference; zero means they match within the lesson tolerance. A signed difference is not the raw derivative. Schedule and recording tables show their actual recorded values instead."
    static func project(_ projection: GradientFormativeProjection?, task: GradientLearningTask, mode: LearningMode) -> Self? {
        guard mode != .assessment, let projection, projection.fixtures.count == 4,
              Set(projection.fixtures.map(\.fixtureID)).count == 4 else { return nil }
        var fixtures: [Fixture] = []
        for (index, fixture) in projection.fixtures.enumerated() {
            guard let shapes = try? task.measuredShapes(widths: GradientLearningTask.widths[index]),
                  Set(fixture.tensors.keys) == Set(shapes.keys) else { return nil }
            var tables: [Table] = []
            for key in fixture.tensors.keys.sorted() {
                guard let measured = fixture.tensors[key], measured.wellFormed, measured.shape == shapes[key] else { return nil }
                let reference: LearningTensor, meaning: String
                if key == "windowSchedule" {
                    reference = .init(shape: [4, 4], values: [1,2,4,1,1,1,1,2,2,2,4,3,2,1,1,4])
                    meaning = "Actual schedule columns: pass, small groups combined, observations included, completed adjustment number. Each row is one adjustment window; the short final window still counts."
                } else if key == "recordingAvailability" {
                    reference = .init(shape: [3], values: [1,0,-1])
                    meaning = "Actual recording checks in order: 1 means a nonzero input effect was measured; 0 means a supplied zero influence was observed; −1 means recording was disabled and the derivative was unavailable, not zero."
                } else {
                    reference = .init(shape: measured.shape, values: Array(repeating: 0, count: measured.values.count))
                    meaning = "Each entry is measured minus reference for this check. Zero means agreement within the lesson tolerance; signed differences are not raw weights or derivatives."
                }
                tables.append(.init(key: key, title: GradientLessonReference.label(key), meaning: meaning,
                                    measured: measured, reference: reference))
            }
            fixtures.append(.init(id: fixture.fixtureID, tables: tables))
        }
        return .init(fixtures: fixtures)
    }
}
struct GradientLearningResultsView: View {
    let presentation: GradientLearningResultsPresentation
    @State private var selectedFixture = ""
    @State private var selectedCheck = ""
    @State private var compareReference = false
    private var fixture: GradientLearningResultsPresentation.Fixture? {
        presentation.fixtures.first { $0.id == selectedFixture } ?? presentation.fixtures.first
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(GradientLearningResultsPresentation.introduction).font(.caption).foregroundStyle(MLTheme.muted)
            FlatChoiceControl(title: "Supplied model and observations", selection: fixture?.id ?? "",
                options: presentation.fixtures.enumerated().map { .init(id: $0.element.id, title: "Example \($0.offset + 1)") }) { selectedFixture = $0 }
            if let fixture {
                let table = fixture.tables.first { $0.key == selectedCheck } ?? fixture.tables.first
                FlatChoiceControl(title: "Measured check", selection: table?.key ?? "",
                    options: fixture.tables.map { .init(id: $0.key, title: $0.title) }) { selectedCheck = $0 }
                Toggle("Compare with independent reference", isOn: $compareReference).toggleStyle(.checkbox).font(.caption)
                if let table {
                    Text(table.meaning).font(.caption).foregroundStyle(MLTheme.muted)
                    LearningTensorTable(tensor: table.measured, title: table.title + " · measured", signed: true)
                        .id(fixture.id + ":" + table.key + ":measured")
                    if compareReference {
                        LearningTensorTable(tensor: table.reference, title: "Independent reference", signed: true)
                            .id(fixture.id + ":" + table.key + ":reference")
                    }
                }
            }
        }.padding(.top, 8)
    }
}
