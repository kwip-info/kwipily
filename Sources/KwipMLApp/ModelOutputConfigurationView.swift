import SwiftUI
import KwipMLCore

/// One draft and one optimistic transaction; stale selections cannot apply to a new node.
struct ModelOutputConfigurationView: View {
    @ObservedObject var context: WorkflowModelNodeContext
    @State private var original: [String: JSONValue] = [:]
    @State private var draft = try! ModelOutputConfiguration([:])
    @State private var sizes: [String] = []
    @State private var error: String?
    @Environment(\.colorScheme) private var colorScheme
    private var accent: Color { colorScheme == .dark ? ComponentVisualFamily.boundaries.primary : ComponentVisualFamily.boundaries.secondary }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            FlatChoiceControl(title:"Output task",selection:draft.task,options:[.init(id:"single",title:"Single output"),.init(id:"gamePolicyValue",title:"Game policy/value")]) {draft.task=$0}
            if draft.task == "gamePolicyValue" {
                Text("Policy · float32 [batch, actions] raw logits\nValue · float32 [batch, 1] in −1…1, for the acting player").font(.caption).foregroundStyle(MLTheme.muted)
                Text("Connect explicit branches to Policy and Value. Use Tanh on the value branch. The shared trunk remains one model; no hidden head or reshape is added.").font(.caption)
            } else {
            Toggle("Check expected output (optional)", isOn: $draft.checkExpectedOutput)
                .toggleStyle(FileInputCheckboxStyle(tint: accent))
            if draft.checkExpectedOutput {
                choice("Expected value type", value: $draft.expectedDType, options: ModelOutputConfiguration.supportedDTypes)
                choice("Expected sample rank", value: $draft.expectedSampleRank, options: ["any", "exact"])
                if draft.expectedSampleRank == "exact" {
                    VStack(spacing: 8) {
                        ForEach(draft.expectedSampleDimensions.indices, id: \.self) { index in
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Text("Sample dimension \(index + 1)").font(.caption)
                                    Spacer()
                                    Button { draft.expectedSampleDimensions.remove(at: index); sizes.remove(at: index) } label: { Image(systemName: "minus") }.buttonStyle(.plain)
                                }
                                HStack {
                                    FlatChoiceControl(title: "Match", selection: draft.expectedSampleDimensions[index] == nil ? "any" : "exact", options: [.init(id: "any", title: "Any"), .init(id: "exact", title: "Exact")], accent: accent) { mode in
                                        draft.expectedSampleDimensions[index] = mode == "any" ? nil : (Int(sizes[index]) ?? 1)
                                        if mode == "exact", sizes[index].isEmpty { sizes[index] = "1" }
                                    }
                                    if draft.expectedSampleDimensions[index] != nil {
                                        TextField("Positive size", text: $sizes[index]).flatField()
                                    }
                                }
                            }
                        }
                        Button("Add dimension") { draft.expectedSampleDimensions.append(nil); sizes.append("") }
                            .buttonStyle(FlatButtonStyle()).disabled(draft.expectedSampleDimensions.count >= 7)
                    }
                }
            }
            }
            if let error { Text(error).font(.caption).foregroundStyle(MLTheme.rose) }
            HStack {
                Button("Apply") { apply() }.disabled(!changed)
                Button("Revert") { load() }.disabled(!changed)
            }.buttonStyle(FlatButtonStyle())
        }.disabled(context.isBusy || !context.isCurrent)
            .onAppear { load() }
            .id(context.contextIdentity)
    }
    private var changed: Bool { draft.values != (try? ModelOutputConfiguration(original).values) || sizes != draft.expectedSampleDimensions.map { $0.map(String.init) ?? "" } }
    private func choice(_ title: String, value: Binding<String>, options: [String]) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption)
            FlatChoiceControl(title: title, selection: value.wrappedValue, options: options.map { .init(id: $0, title: $0 == "any" ? "Any" : $0 == "exact" ? "Exact" : $0) }, accent: accent) { value.wrappedValue = $0 }
        }
    }
    private func load() {
        original = context.configuration
        do { draft = try ModelOutputConfiguration(original); sizes = draft.expectedSampleDimensions.map { $0.map(String.init) ?? "" }; error = nil }
        catch { self.error = error.localizedDescription }
    }
    private func apply() {
        do {
            var next = draft
            for index in next.expectedSampleDimensions.indices where next.expectedSampleDimensions[index] != nil {
                guard let size = Int(sizes[index]), (1...100_000_000).contains(size) else { throw GraphConfigurationError("Enter a positive whole size for sample dimension \(index + 1).") }
                next.expectedSampleDimensions[index] = size
            }
            _ = try ModelOutputConfiguration(next.values)
            guard context.authority.configure(next.values, expected: original) else { error = context.statusMessage; return }
            load()
        } catch { self.error = error.localizedDescription }
    }
}
