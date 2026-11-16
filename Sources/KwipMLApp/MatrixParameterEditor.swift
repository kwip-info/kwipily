import KwipMLCore
import SwiftUI

/// Parameter values are learner-authored; this view never derives or inserts a solution.
struct MatrixParameterEditor: View {
  @ObservedObject var lesson: AffineLearningStore
  @ObservedObject var workspace: WorkspaceStore
  let node: GraphNode
  @State private var rowPage = 0
  @State private var columnPage = 0
  @State private var confirmReset = false

  private var biasEnabled: Bool { node.config["bias"]?.boolValue != false }
  private var parameters: LinearParameterValues? { lesson.parameters(for: node.id) }
  private var matchesShape: Bool {
    guard let values = parameters else { return false }
    return values.weight.shape.map(Double.init)
      == [node.config["out_features"]?.doubleValue, node.config["in_features"]?.doubleValue]
      .compactMap { $0 }
      && values.bias.shape.map(Double.init)
        == [node.config["out_features"]?.doubleValue].compactMap { $0 }
  }
  private var origin: String {
    "\(workspace.activeProjectID ?? ""):\(node.id):\(node.config["in_features"] ?? 0):\(node.config["out_features"] ?? 0):\(lesson.parameterRevision)"
  }
  private func currentOrigin() -> String? {
    guard workspace.selectedNodeID == node.id, workspace.selectedNode?.config == node.config,
      !lesson.isBusy, matchesShape
    else { return nil }
    return origin
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      SectionLabel(title: "Explicit parameters")
      Text(biasEnabled ? "W[out, in] · b[out]" : "W[out, in] · no bias parameter").font(
        .system(size: 12, design: .monospaced)
      ).foregroundStyle(MLTheme.teal)
      Text(
        "Columns index inputs; rows index outputs. Indices below start at 0. Values must be finite within ±1,000,000."
      ).font(.caption).foregroundStyle(MLTheme.muted)
      if let values = parameters, matchesShape, values.bias.wellFormed,
        let page = MatrixPage(
          tensor: values.weight, rowPage: rowPage, columnPage: columnPage, pageSize: 2)
      {
        let revision = lesson.parameterRevision
        HStack {
          SectionLabel(title: "Weight matrix")
          Spacer()
          Text("\(values.weight.shape[0]) × \(values.weight.shape[1])").font(.caption.monospaced())
        }
        ForEach(Array(page.rows), id: \.self) { row in
          HStack(alignment: .top, spacing: 8) {
            ForEach(Array(page.columns), id: \.self) { column in
              VStack(alignment: .leading, spacing: 4) {
                Text("W[\(row), \(column)]").font(.system(size: 10, design: .monospaced))
                  .foregroundStyle(MLTheme.muted)
                BoundedNumberField(
                  title: "Weight row \(row), column \(column)",
                  value: .decimal(page.value(row: row, column: column) ?? 0),
                  domain: .decimal(minimum: -1_000_000, maximum: 1_000_000),
                  origin: origin + ":w:\(row):\(column)",
                  currentOrigin: { currentOrigin().map { $0 + ":w:\(row):\(column)" } },
                  onDraftChange: { lesson.setNumericEditPending(id: $0, pending: $1) }
                ) {
                  lesson.setWeight(
                    nodeID: node.id, row: row, column: column, value: $0.double, revision: revision)
                }.font(.system(size: 11)).frame(maxWidth: .infinity)
              }
            }
          }
        }
        if page.columnPageCount > 1 {
          paging("Input columns", page: $columnPage, count: page.columnPageCount)
        }
        SectionLabel(title: biasEnabled ? "Bias" : "Stored bias · inactive")
        ForEach(Array(page.rows), id: \.self) { row in
          HStack {
            Text("b[\(row)]").font(.system(size: 11, design: .monospaced)).frame(
              width: 42, alignment: .leading)
            BoundedNumberField(
              title: "Bias index \(row)", value: .decimal(values.bias.values[row]),
              domain: .decimal(minimum: -1_000_000, maximum: 1_000_000),
              origin: origin + ":b:\(row)",
              currentOrigin: { currentOrigin().map { $0 + ":b:\(row)" } },
              onDraftChange: { lesson.setNumericEditPending(id: $0, pending: $1) }
            ) {
              lesson.setBias(nodeID: node.id, index: row, value: $0.double, revision: revision)
            }.font(.system(size: 11)).disabled(!biasEnabled)
          }
        }
        if page.rowPageCount > 1 { paging("Output rows", page: $rowPage, count: page.rowPageCount) }
      } else {
        Text(
          parameters == nil
            ? "No explicit values yet. Initialize zeros, then enter your weights and bias."
            : "Dimensions changed. Stored values remain preserved, but cannot be submitted with these dimensions. Undo the shape change or explicitly reset this block."
        )
        .font(.system(size: 12)).foregroundStyle(MLTheme.rose)
      }
      Text("Apply edits before checking. Leaving this editor discards any unapplied field drafts.")
        .font(.system(size: 10)).foregroundStyle(MLTheme.muted)
      if confirmReset {
        Text(
          "Replace this block’s stored weights and bias with zeros? This does not change its dimensions."
        ).font(.caption)
        HStack {
          Button("Replace with zeros") {
            lesson.resetParameters(nodeID: node.id)
            confirmReset = false
            rowPage = 0
            columnPage = 0
          }
          Button("Keep values") { confirmReset = false }
        }.buttonStyle(FlatButtonStyle())
      } else {
        Button(parameters == nil ? "Initialize zeros" : "Reset parameters…") {
          if parameters == nil {
            lesson.resetParameters(nodeID: node.id)
          } else {
            confirmReset = true
          }
        }.buttonStyle(FlatButtonStyle())
      }
    }.disabled(lesson.isBusy)
      .onChange(of: node.id) { _, _ in
        rowPage = 0
        columnPage = 0
        confirmReset = false
      }
  }
  private func paging(_ title: String, page: Binding<Int>, count: Int) -> some View {
    HStack {
      Text(title).font(.caption)
      Spacer()
      Button {
        page.wrappedValue = max(0, min(page.wrappedValue, count - 1) - 1)
      } label: {
        Image(systemName: "chevron.left")
      }
      .disabled(page.wrappedValue <= 0).accessibilityLabel("Previous \(title)")
      Text("\(min(page.wrappedValue, count - 1) + 1) / \(count)").font(.caption.monospaced())
      Button {
        page.wrappedValue = min(count - 1, page.wrappedValue + 1)
      } label: {
        Image(systemName: "chevron.right")
      }
      .disabled(page.wrappedValue >= count - 1).accessibilityLabel("Next \(title)")
    }.buttonStyle(.borderless)
  }
}
