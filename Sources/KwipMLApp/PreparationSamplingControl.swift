import SwiftUI
import KwipMLCore

struct PreparationSamplingControl: View {
    @ObservedObject var store: WorkspaceStore
    var compact = false
    @State private var count = "1000"
    @State private var percent = "10"
    @State private var error: String?
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if compact {
                FlatChoiceControl(title: "Working data", selection: store.preparationSampling.mode.rawValue,
                    options: [.init(id: "all", title: "All data"), .init(id: "rowCount", title: "Record count"), .init(id: "percentage", title: "Percentage")]) { selected in
                    switch PreparationSampling.Mode(rawValue: selected) {
                    case .all: apply(.all)
                    case .rowCount: apply(.init(mode: .rowCount, rowCount: Int(count)))
                    case .percentage: apply(.init(mode: .percentage, percentage: Double(percent)))
                    case nil: break
                    }
                }
            } else {
            SectionLabel(title: "Working data")
            HStack(spacing: 6) {
                modeButton("All data", mode: .all)
                modeButton("Records", mode: .rowCount)
                modeButton("Percent", mode: .percentage)
            }
            }
            if store.preparationSampling.mode == .rowCount {
                TextField("Records", text: $count).flatField().onSubmit { apply(.init(mode: .rowCount, rowCount: Int(count))) }
                Button("Apply record count") { apply(.init(mode: .rowCount, rowCount: Int(count))) }.buttonStyle(FlatButtonStyle())
            } else if store.preparationSampling.mode == .percentage {
                TextField("Percent", text: $percent).flatField().onSubmit { apply(.init(mode: .percentage, percentage: Double(percent))) }
                Button("Apply percentage") { apply(.init(mode: .percentage, percentage: Double(percent))) }.buttonStyle(FlatButtonStyle())
            }
            if !compact || store.preparationSampling.mode != .all { Text(store.preparationSampling.title).font(.caption.weight(.semibold)) }
            if !compact { Text("Applies when all data is not required. View, file outputs and training use complete upstream data. Samples take the first source records—files for File Input. One file can produce multiple rows. Percentage may require counting the source.").font(.caption2).foregroundStyle(MLTheme.muted).fixedSize(horizontal: false, vertical: true) }
            if let error { Text(error).font(.caption).foregroundStyle(MLTheme.rose) }
        }.padding(compact ? 0 : 12).disabled(store.isBusy)
        .onAppear { synchronize() }
        .onChange(of:store.preparationSampling) {_,_ in synchronize() }
    }
    // Inline choices keep this toolbar panel open while editing the budget.
    // A second attached dropdown would resign and close the containing panel.
    private func modeButton(_ title: String, mode: PreparationSampling.Mode) -> some View {
        Button(title) {
            switch mode {
            case .all: store.preparationSampling = .all; error = nil
            case .rowCount: apply(.init(mode: .rowCount, rowCount: Int(count)))
            case .percentage: apply(.init(mode: .percentage, percentage: Double(percent)))
            }
        }
        .buttonStyle(FlatButtonStyle(prominent: store.preparationSampling.mode == mode))
        .accessibilityLabel("Working data: \(title)")
        .accessibilityValue(store.preparationSampling.mode == mode ? "Selected" : "")
    }
    private func synchronize() {
        if let appliedCount=store.preparationSampling.rowCount {count=String(appliedCount)}
        if let appliedPercent=store.preparationSampling.percentage {percent=String(appliedPercent)}
    }
    private func apply(_ policy: PreparationSampling) {
        do { store.preparationSampling = try policy.validated(); error = nil }
        catch { self.error = error.localizedDescription }
    }
}
