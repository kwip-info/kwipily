import SwiftUI
import KwipMLCore

/// Individual editable values. Commas remain an internal bridge to the established typed parser.
struct ExperimentAxisValuesDraft: Equatable {
    private(set) var values: [String: [String]] = [:]
    init(specification: WorkflowExperimentSpecification? = nil) {
        for axis in specification?.axes ?? [] { values[axis.field] = axis.values.map(Self.text) }
    }
    static func text(_ value: JSONValue) -> String {
        switch value { case .integer(let n): return String(n); case .number(let n): return String(n); case .string(let s): return s; default: return "" }
    }
    mutating func add(_ key: String) {
        guard WorkflowControlContracts.experimentFields.contains(where: { $0.key == key }), (values[key]?.count ?? 0) < 32 else { return }
        values[key, default: []].append("")
    }
    mutating func set(_ text: String, key: String, index: Int) {
        guard values[key]?.indices.contains(index) == true else { return }
        values[key]![index] = String(text.prefix(256))
    }
    mutating func remove(_ key: String, index: Int) {
        guard values[key]?.indices.contains(index) == true else { return }
        values[key]!.remove(at: index)
        if values[key]?.isEmpty == true { values.removeValue(forKey: key) }
    }
    func validate() throws {
        for (key, tokens) in values {
            guard WorkflowControlContracts.experimentFields.contains(where: { $0.key == key }), tokens.count <= 32 else { throw DataFlowContractError("Choose a supported setting with at most 32 values.") }
            guard tokens.allSatisfy({ !$0.contains(",") && !$0.contains("\n") && !$0.contains("\r") }) else {
                throw DataFlowContractError("Enter one value in each field. Use Add value for another option.")
            }
            guard tokens.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) else {
                throw DataFlowContractError("Complete or remove each added value.")
            }
        }
    }
    func applying(to draft: WorkflowExperimentDraft) -> WorkflowExperimentDraft {
        var result = draft
        result.values = values.mapValues { $0.joined(separator: ",") }
        return result
    }
}

struct ExperimentAxisValuesView: View {
    let title: String
    let values: [JSONValue]
    @Environment(\.colorScheme) private var colorScheme
    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Text(title).font(.caption.weight(.medium)).frame(width: 104, alignment: .leading)
            ExperimentValueFlow {
                ForEach(Array(values.enumerated()), id: \.offset) { _, value in
                    Text(ExperimentAxisValuesDraft.text(value)).font(.caption.monospaced()).fixedSize()
                        .padding(.horizontal, 7).padding(.vertical, 4)
                        .background(ComponentVisualFamily.training.primary.opacity(colorScheme == .dark ? 0.15 : 0.09))
                        .overlay(Rectangle().stroke(ComponentVisualFamily.training.primary.opacity(0.3)))
                }
            }
        }.padding(.vertical, 7)
    }
}

/// Tiny bounded chip flow: values keep their full text and wrap between chips.
private struct ExperimentValueFlow: Layout {
    private func positions(_ subviews: Subviews, width: CGFloat) -> (points: [CGPoint], height: CGFloat) {
        var x: CGFloat = 0, y: CGFloat = 0, row: CGFloat = 0
        var points: [CGPoint] = []
        for view in subviews {
            let size = view.sizeThatFits(.unspecified)
            if x > 0 && x + size.width > width { x = 0; y += row + 5; row = 0 }
            points.append(.init(x: x, y: y)); x += size.width + 5; row = max(row, size.height)
        }
        return (points, y + row)
    }
    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? 300
        return .init(width: width, height: positions(subviews, width: width).height)
    }
    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let result = positions(subviews, width: bounds.width)
        for (index, view) in subviews.enumerated() {
            view.place(at: .init(x: bounds.minX + result.points[index].x, y: bounds.minY + result.points[index].y), anchor: .topLeading, proposal: .unspecified)
        }
    }
}
