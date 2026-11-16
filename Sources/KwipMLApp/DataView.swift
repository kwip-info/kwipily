import SwiftUI
struct DataView: View {
    @ObservedObject var store: WorkspaceStore
    @ObservedObject var session: PreparationSession
    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 14) {
                Button("Source & model mapping") { session.showPreparation = false }
                    .foregroundStyle(session.showPreparation ? MLTheme.muted : MLTheme.rose)
                Button("Preparation canvas") {
                    if store.activeProjectID != nil || store.saveManagedProject() { _ = session.preparation(for: store); session.showPreparation = true }
                }.foregroundStyle(session.showPreparation ? MLTheme.rose : MLTheme.muted)
                Spacer()
            }.buttonStyle(.borderless).font(.system(size: 12, weight: .medium)).padding(12).background(MLTheme.panel)
            Divider()
            if session.showPreparation, let preparation = session.current { DataFlowWorkspaceView(store: preparation).id(store.activeProjectID) }
            else { sourceMapping }
        }.task(id: store.activeProjectID) {
            session.attach(store)
            if session.showPreparation { _ = session.preparation(for: store) }
        }
    }
    private var sourceMapping: some View {
        ThemedScrollView {
            VStack(alignment: .leading, spacing: 22) {
                Text("Data workspace").font(.system(size: 28, weight: .semibold))
                Text("Source → Preview → Map columns → Split → Model input").font(.system(size: 13)).foregroundStyle(MLTheme.teal)
                Text("Prepare data separately from the architecture. The source adapter produces the same typed batch contract for the model.").foregroundStyle(MLTheme.muted)
                Divider()
                SectionLabel(title: "Source")
                HStack { Button("Bundled sample") { store.useSampleData() }; Button("Choose local CSV…") { store.chooseCSV() }; Spacer(); if store.isInspectingData { ProgressView().controlSize(.small) } }.disabled(store.isRunning || store.isPreviewing)
                if let prepared = store.execution.preparedTraining {
                    Text("Prepared snapshot selected").font(.headline)
                    Text("\(prepared.trainingNode) · \(prepared.trainingPort) → training\n\(prepared.validationNode) · \(prepared.validationPort) → validation").font(.system(size: 12, design: .monospaced))
                    Text("Features: \(prepared.featureColumns.joined(separator: ", ")) · Target: \(prepared.targetColumn)").font(.caption)
                    Text("Training uses the exact saved partitions and fitted transforms. To change the split or source, prepare a new snapshot and select its outputs. Bundled sample or Choose local CSV switches back to direct data.").font(.caption).foregroundStyle(MLTheme.muted)
                } else if let csv = store.execution.datasetCSV {
                    Text(URL(fileURLWithPath: csv).lastPathComponent).font(.headline)
                    Text(csv).font(.caption).foregroundStyle(MLTheme.muted).textSelection(.enabled)
                    HStack { Text("\(store.dataRowCount) rows · \(store.dataColumns.count) columns").font(.caption); Button("Refresh preview") { store.inspectData() }.disabled(store.isInspectingData || store.isPreviewing || store.isRunning) }
                    if let error = store.dataError { Text(error).foregroundStyle(.orange).font(.system(size: 12)) }
                    if !store.dataColumns.isEmpty {
                        Divider(); SectionLabel(title: "Column mapping")
                        VStack(alignment: .leading, spacing: 14) {
                            columnPicker("Feature 1", index: 0)
                            columnPicker("Feature 2", index: 1)
                            VStack(alignment: .leading) {
                                Text("Target").font(.caption)
                                columnMenu(title:"Target column",value:store.execution.targetColumn ?? "") { value in store.setExecution { $0.targetColumn = value } }

                            }
                        }.disabled(store.isRunning || store.isPreviewing)
                        Text("Select two distinct numeric features and a separate binary target (0 or 1). Invalid values are reported before training; no silent dropping or coercion.").font(.caption).foregroundStyle(MLTheme.muted)
                        Divider(); SectionLabel(title: "Source preview · first 8 rows")
                        ThemedScrollView(.horizontal) {
                            Grid(alignment: .leading, horizontalSpacing: 24, verticalSpacing: 10) {
                                GridRow { ForEach(store.dataColumns, id: \.self) { Text($0).font(.system(size: 11, weight: .semibold)).foregroundStyle(MLTheme.teal) } }
                                ForEach(Array(store.dataRows.enumerated()), id: \.offset) { _, row in GridRow { ForEach(Array(row.enumerated()), id: \.offset) { _, cell in Text(cell).font(.system(size: 11, design: .monospaced)).lineLimit(1) } } }
                            }.textSelection(.enabled)
                        }
                    }
                } else {
                    Text("Two moons · bundled sample").font(.headline)
                    Text("1,000 available synthetic observations · 2 numeric features · 2 classes").font(.system(size: 13)).foregroundStyle(MLTheme.muted)
                    Text("No downloads or account needed. A fixed source keeps model initialization and data selection independently reproducible.").font(.caption).foregroundStyle(MLTheme.muted)
                }
                if store.runtime?.kind == .bundled { Text("Selected sources are copied into private local snapshots for execution. File permissions stay on this Mac; portable project files contain references, not permission grants.").font(.caption).foregroundStyle(MLTheme.muted) }
                Divider(); SectionLabel(title: "Preparation contract")
                Text("Direct sources use the sample limit and reproducible stratified split in Settings. The preparation canvas instead defines its own partitions and fits missing-value replacements, categorical vocabularies and scaling statistics on training rows, then applies the same fitted state to both partitions. Select its exact outputs for training. Every completed run retains row identities and preparation provenance. Joins, remote connectors and additional source adapters remain planned.").font(.system(size: 12)).foregroundStyle(MLTheme.muted)
                Button("Open model diagram") { store.activeSection = "Build" }
            }.padding(28).buttonStyle(FlatButtonStyle())
        }
    }
    private func columnPicker(_ title: String, index: Int) -> some View {
        VStack(alignment: .leading) {
            Text(title).font(.caption)
            columnMenu(title:title+" column",value:store.execution.featureColumns.indices.contains(index) ? store.execution.featureColumns[index] : "") { value in
                store.setExecution { setting in while setting.featureColumns.count < 2 { setting.featureColumns.append("") }; setting.featureColumns[index] = value }
            }

        }
    }
    private func columnMenu(title:String,value:String,select:@escaping(String)->Void)->some View {
        FlatChoiceControl(title: title, selection: value,
                          options: [FlatChoiceOption(id: "", title: "Clear selection")] + store.dataColumns.map { FlatChoiceOption(id: $0, title: $0) },
                          placeholder: "Select column", onSelect: select)
            .frame(maxWidth: 430)
    }

}
