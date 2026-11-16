import SwiftUI
import Combine
import KwipMLCore

/// One exact source baseline shared by the six retained role editors.
@MainActor final class TrainInlineRecipesSession: ObservableObject {
    static let roles = ["optimizer", "scheduler", "objective", "initialization", "gradientPolicy", "trainability"]
    let authority: WorkflowNodeEditingAuthority
    private(set) var expected: [String: JSONValue]
    private var observation: AnyCancellable?
    init(authority: WorkflowNodeEditingAuthority) {
        self.authority = authority; expected = authority.configuration
        observation = authority.objectWillChange.sink { [weak self] _ in
            self?.synchronizeOwnChanges(); self?.objectWillChange.send()
        }
    }
    private func synchronizeOwnChanges() {
        guard authority.isCurrent else { return }
        let current = authority.configuration
        // The shared authority stays alive only for its own commands; external changes retire it.
        expected = current
    }
    var isCurrent: Bool { authority.isCurrent && authority.configuration == expected }
    var isBusy: Bool { !isCurrent || authority.isBusy }
    @discardableResult func apply(role: String, value: JSONValue?, objectiveScalarization: String? = nil) -> Bool {
        guard Self.roles.contains(role), !isBusy else { return false }
        var next = expected
        next[role] = value
        if let scalarization = objectiveScalarization {
            guard role == "objective", ["mean", "sum"].contains(scalarization) else { return false }
            next["objectiveScalarization"] = .string(scalarization)
        }
        do {
            _ = try WorkflowTrainRecipes(configuration: next, schemaVersion: max(13, authority.owner.document.schemaVersion))
            guard authority.configure(next, expected: expected) else { return false }
            expected = authority.configuration
            objectWillChange.send(); return true
        } catch { authority.message = error.localizedDescription; return false }
    }
    func modelChoices() -> (endpoints: [GradientEndpointChoice], owners: [InitializationOwnerChoice]) {
        guard isCurrent else { return ([], []) }
        let owner = authority.owner, document = owner.document
        let edges = document.edges.filter { $0.to.address == authority.address && $0.to.port == "model" }
        guard edges.count == 1, let definition = edges.first?.from.address else { return ([], []) }
        let context: GraphValidationContext = expected["objective"] == nil ? .preparedWorkflow : .explicitObjective
        guard let model = try? WorkflowCompiler.modelSnapshot(document, definition: definition,
            modelRegistry: owner.modelRegistry, dataRegistry: owner.dataRegistry, context: context) else { return ([], []) }
        let validation = GraphValidator.validate(model.graph, registry: owner.modelRegistry, context: context)
        guard validation.isValid else { return ([], []) }
        let layerNames=TrainLayerPresentation.names(for:model.graph,origins:model.origins,document:document,registry:owner.modelRegistry)
        let owners = ((try? WorkflowLifecycleVerifier.optimizerParameterOwnership(model.graph, context: context,
            capturedRegistry: owner.modelRegistry, initializationOwnership: true)) ?? []).compactMap { raw -> InitializationOwnerChoice? in
            guard case .object(let row) = raw, let node = row["node"]?.stringValue,
                  let parameter = row["parameter"]?.stringValue,
                  case .array(let shape) = row["shape"] else { return nil }
            let dimensions = shape.compactMap { $0.integerValue.flatMap(Int.init(exactly:)) }
            guard dimensions.count == shape.count else { return nil }
            return .init(node: node, parameter: parameter, name: layerNames[node] ?? "Layer", shape: dimensions)
        }
        var endpoints: [GradientEndpointChoice] = []
        for node in model.graph.nodes {
            guard let definition = validation.configuredDefinition(nodeID: node.id, graph: model.graph, registry: owner.modelRegistry) else { continue }
            for port in definition.outputs {
                guard let contract = validation.outputTensorContracts[.init(node.id, port)] else { continue }
                endpoints.append(.init(node: node.id, port: port, name: layerNames[node.id] ?? (node.name.isEmpty ? definition.title : node.name),
                    dtype: contract.dtype, shape: contract.layout.sample.dimensions, isModelInput: ["input", "index_input"].contains(node.type)))
            }
        }
        return (endpoints, owners)
    }
}

/// One role's typed draft. Other role Applies renew the shared baseline without discarding this draft.
@MainActor final class TrainInlineRecipeDraft: ObservableObject {
    let session: TrainInlineRecipesSession
    let role: String
    @Published private(set) var componentID: String? = nil
    @Published private(set) var configuration: [String: JSONValue] = [:]
    @Published private(set) var objectiveScalarization: String = "mean"
    private var baselineScalarization: String = "mean"
    @Published private(set) var resetGeneration = 0
    @Published private(set) var rejection: String?
    private var baseline: JSONValue?
    private var numericPending = Set<String>()
    private var dormantConfigurations: [String: [String: JSONValue]] = [:]
    private var observations: [AnyCancellable] = []
    private(set) var initialization: InitializationRecipeInspectorContext?
    private(set) var gradient: GradientPolicyInspectorContext?
    private(set) var trainability: TrainabilityInspectorContext?
    private var childObservation: AnyCancellable?
    init(session: TrainInlineRecipesSession, role: String) {
        self.session = session; self.role = role; baseline = session.expected[role]
        objectiveScalarization = session.expected["objectiveScalarization"]?.stringValue ?? "mean"
        baselineScalarization = objectiveScalarization
        if case .object(let envelope) = baseline {
            componentID = envelope["componentID"]?.stringValue
            if case .object(let values) = envelope["config"] { configuration = values }
        }
        rebuildStructuredEditor()
        observations.append(session.objectWillChange.sink { [weak self] _ in
            self?.synchronizeRoleBaseline(); self?.objectWillChange.send()
        })
    }
    private func synchronizeRoleBaseline() {
        guard session.isCurrent else { return }
        let scalarization = session.expected["objectiveScalarization"]?.stringValue ?? "mean"
        guard baseline != session.expected[role] || (role == "objective" && baselineScalarization != scalarization) else { return }
        // Own mapping commits may update a role. Refresh clean editors; retain
        // conflicting drafts until explicit Revert rather than overwriting them.
        guard !hasChanges, !hasUncommittedNumbers else { return }
        baseline = session.expected[role]; baselineScalarization = scalarization
        _ = revert()
    }
    static func IDs(_ role: String) -> [String] {
        switch role {
        case "optimizer": return OptimizerConfiguration.componentIDs.sorted()
        case "scheduler": return SchedulerConfiguration.componentIDs.sorted()
        case "objective": return LossConfiguration.componentIDs.sorted()
        case "initialization": return ["initialization"]
        case "gradientPolicy": return ["gradient_policy"]
        case "trainability": return ["trainability"]
        default: return []
        }
    }
    static func title(_ role: String) -> String {
        ["optimizer": "Optimizer", "scheduler": "Schedule", "objective": "Objective", "initialization": "Starting values",
         "gradientPolicy": "Gradients", "trainability": "Trainability"][role] ?? role
    }
    static func componentTitle(_ id: String) -> String {
        if let title = ["initialization": "Custom starting values", "gradient_policy": "Custom gradients", "trainability": "Custom learned-array behavior"][id] { return title }
        return BundledOptimizerComponents.entries[id]?.definition.title ?? BundledSchedulerComponents.entries[id]?.definition.title ??
        BundledLossComponents.entries[id]?.definition.title ?? ["initialization":"Custom starting values","gradient_policy":"Custom gradients","trainability":"Custom trainability"][id] ?? id
    }
    var fields: [ComponentField] {
        guard let id = componentID else { return [] }
        return BundledOptimizerComponents.entries[id]?.definition.fields ?? BundledSchedulerComponents.entries[id]?.definition.fields ?? BundledLossComponents.entries[id]?.definition.fields ?? []
    }
    var isBusy: Bool { session.isBusy }
    var hasUncommittedNumbers: Bool { !numericPending.isEmpty || initialization?.hasUncommittedNumbers == true || gradient?.hasUncommittedNumbers == true }
    private var typedValues: [String: JSONValue] {
        initialization?.draftConfiguration ?? gradient?.draftConfiguration ?? trainability?.draftConfiguration ?? configuration
    }
    var hasChanges: Bool {
        if role == "objective", componentID != nil, objectiveScalarization != baselineScalarization { return true }
        guard let id = componentID else { return baseline != nil }
        guard case .object(let envelope) = baseline else { return true }
        return envelope["componentID"] != .string(id) || envelope["config"] != .object(typedValues) || initialization?.hasChanges == true
    }
    var canApply: Bool { !isBusy && hasChanges && !hasUncommittedNumbers && validationMessage == nil }
    var validationMessage: String? {
        guard baseline == session.expected[role], role != "objective" || baselineScalarization == (session.expected["objectiveScalarization"]?.stringValue ?? "mean") else { return "This setting changed. Revert to load its saved value; your draft is preserved." }
        do { _ = try value(); return nil } catch { return error.localizedDescription }
    }
    func numericOrigin(_ key: String) -> String { session.authority.numericOrigin(key: "inline.\(role).\(resetGeneration).\(key)") }
    func currentNumericOrigin(_ key: String) -> String? { isBusy ? nil : numericOrigin(key) }
    private var pendingKey: String { "inlineRecipeDraft.\(role)" }
    private func reportPending(_ child: Bool = false) {
        let authority = session.authority
        authority.setNumericPending(origin: authority.numericOrigin(key: pendingKey), key: pendingKey,
                                    pending: child || hasChanges || hasUncommittedNumbers)
        objectWillChange.send()
    }
    func setNumericPending(_ key: String, origin: String, pending: Bool) {
        guard origin == currentNumericOrigin(key) else { return }
        if pending { numericPending.insert(key) } else { numericPending.remove(key) }
        reportPending()
    }
    @discardableResult func choose(_ id: String?) -> Bool {
        guard !isBusy, !hasUncommittedNumbers, id == nil || Self.IDs(role).contains(id!) else { return false }
        if let previous = componentID {
            var retained = typedValues
            if let editor = initialization, let seed = Int64(editor.seedText), seed >= 0 { retained["seed"] = .integer(seed) }
            dormantConfigurations[previous] = retained
        }
        // Keep each family's typed draft when switching methods; Revert alone discards it.
        if let id {
            if let dormant = dormantConfigurations[id] { configuration = dormant }
            else if case .object(let envelope) = baseline, envelope["componentID"] == .string(id), case .object(let saved) = envelope["config"] {
                configuration = saved
            } else {
                do { configuration = try Self.defaults(role: role, id: id) }
                catch { rejection = error.localizedDescription; return false }
            }
        } else { configuration = [:] }
        componentID = id; resetGeneration += 1; rejection = nil
        rebuildStructuredEditor(); reportPending(); return true
    }
    static func defaults(role: String, id: String) throws -> [String: JSONValue] {
        switch role {
        case "optimizer": return try OptimizerConfiguration(componentID: id, values: [:]).values
        case "scheduler": return try SchedulerConfiguration(componentID: id, values: [:]).values
        case "objective": return try LossConfiguration(componentID: id, values: [:]).values
        case "initialization": return ["seed": .integer(42), "rules": .array([])]
        case "gradientPolicy": return try GradientPolicyConfiguration(values: [:]).values
        case "trainability": return try TrainabilityConfiguration(values: [:]).values
        default: throw GraphConfigurationError("Choose a supported Train setting.")
        }
    }
    @discardableResult func setObjectiveScalarization(_ value: String) -> Bool {
        guard role == "objective", componentID != nil, !isBusy, ["mean", "sum"].contains(value) else { return false }
        objectiveScalarization = value; rejection = nil; reportPending(); return true
    }
    @discardableResult func setField(_ key: String, value: JSONValue, origin: String) -> Bool {
        guard !isBusy, origin == currentNumericOrigin(key), let field = fields.first(where: { $0.key == key }), field.accepts(value) else { return false }
        configuration[key] = value; rejection = nil; reportPending(); return true
    }
    private func rebuildStructuredEditor() {
        initialization?.releasePendingEdits(); gradient?.releasePendingEdits(); trainability?.releasePendingEdits()
        initialization = nil; gradient = nil; trainability = nil; childObservation = nil
        guard componentID != nil else { return }
        let choices = session.modelChoices()
        let current = { [weak self] in self?.session.isCurrent == true }
        let busy = { [weak self] in self?.session.isBusy != false }
        let pending: (Bool) -> Void = { [weak self] value in self?.reportPending(value) }
        // Inline Apply is owned by this role; child editors supply typed values and validation only.
        let unused: ([String: JSONValue], [String: JSONValue]) -> Bool = { _, _ in false }
        let origin = session.authority.contextIdentity + ".inline." + role + "." + String(resetGeneration)
        if role == "initialization" {
            initialization = .init(configuration: configuration, owners: choices.owners, origin: origin,
                isCurrent: current, isBusy: busy, apply: unused, pending: pending)
            childObservation = initialization?.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }
        } else if role == "gradientPolicy" {
            gradient = .init(configuration: configuration, endpoints: choices.endpoints, owners: choices.owners, origin: origin,
                isCurrent: current, isBusy: busy, apply: unused, pending: pending)
            childObservation = gradient?.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }
        } else if role == "trainability" {
            trainability = .init(configuration: configuration, owners: choices.owners, origin: origin,
                isCurrent: current, isBusy: busy, apply: unused, pending: pending)
            childObservation = trainability?.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }
        }
    }
    func value() throws -> JSONValue? {
        guard let id = componentID else { return nil }
        switch role {
        case "optimizer": return try OptimizerConfiguration(componentID: id, values: configuration).recipeValue
        case "scheduler": return try SchedulerConfiguration(componentID: id, values: configuration).recipeValue
        case "objective": return try LossConfiguration(componentID: id, values: configuration).recipeValue
        case "initialization": return try InitializationConfiguration(values: initialization?.validatedConfiguration() ?? configuration).recipeValue
        case "gradientPolicy": return try GradientPolicyConfiguration(values: gradient?.validatedConfiguration() ?? configuration).recipeValue
        case "trainability": return try TrainabilityConfiguration(values: trainability?.validatedConfiguration() ?? configuration).recipeValue
        default: throw GraphConfigurationError("Choose a supported Train setting.")
        }
    }
    @discardableResult func apply() -> Bool {
        guard canApply, baseline == session.expected[role],
              role != "objective" || baselineScalarization == (session.expected["objectiveScalarization"]?.stringValue ?? "mean") else { return false }
        do {
            let updated = try value()
            guard session.apply(role: role, value: updated, objectiveScalarization: role == "objective" && componentID != nil && objectiveScalarization != baselineScalarization ? objectiveScalarization : nil) else { rejection = session.authority.message ?? "The source changed. Reopen this editor."; return false }
            baseline = updated; dormantConfigurations.removeAll()
            baselineScalarization = objectiveScalarization
            if case .object(let envelope) = updated, case .object(let values) = envelope["config"] { configuration = values }
            else { configuration = [:] }
            numericPending.removeAll(); resetGeneration += 1; rejection = nil
            rebuildStructuredEditor(); reportPending(); return true
        } catch { rejection = error.localizedDescription; return false }
    }
    @discardableResult func revert() -> Bool {
        guard !isBusy else { return false }
        baseline = session.expected[role]
        baselineScalarization = session.expected["objectiveScalarization"]?.stringValue ?? "mean"
        componentID = nil; configuration = [:]; dormantConfigurations.removeAll()
        objectiveScalarization = baselineScalarization
        if case .object(let envelope) = baseline {
            componentID = envelope["componentID"]?.stringValue
            if case .object(let values) = envelope["config"] { configuration = values }
        }
        numericPending.removeAll(); resetGeneration += 1; rejection = nil
        rebuildStructuredEditor(); reportPending(); return true
    }
    func releasePendingEdits() {
        initialization?.releasePendingEdits(); gradient?.releasePendingEdits(); trainability?.releasePendingEdits()
        numericPending.removeAll()
        let authority = session.authority
        authority.setNumericPending(origin: authority.numericOrigin(key: pendingKey), key: pendingKey, pending: false)
    }
}

struct TrainInlineRecipesInspectorView: View {
    @StateObject private var session: TrainInlineRecipesSession
    init(authority: WorkflowNodeEditingAuthority) { _session = StateObject(wrappedValue: TrainInlineRecipesSession(authority: authority)) }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            TrainInspectorSection("Optimization",symbol:"slider.horizontal.3",summary:summary(["optimizer","scheduler","objective"])) {
                ForEach((session.expected["fineTuning"] == nil ? ["optimizer","scheduler","objective"]:["optimizer","scheduler"]),id:\.self) {TrainInlineRoleEditor(session:session,role:$0)}
            }
            if session.expected["fineTuning"] == nil {
            TrainInspectorSection("Gradients",symbol:"arrow.triangle.branch",summary:summary(["gradientPolicy"])) {
                TrainInlineRoleEditor(session:session,role:"gradientPolicy")
            }
            }
            if session.expected["initialization"] != nil || session.expected["trainability"] != nil {
                TrainInspectorSection("Legacy layer overrides",symbol:"clock.arrow.circlepath",summary:"Retained") {
                    ForEach(["initialization","trainability"].filter {session.expected[$0] != nil},id:\.self) {TrainInlineRoleEditor(session:session,role:$0)}
                }
            }
        }
    }
    private func summary(_ roles:[String])->String {
        let selected=roles.compactMap {role->String? in
            guard case .object(let recipe)=session.expected[role],let id=recipe["componentID"]?.stringValue else {return nil}
            return TrainInlineRecipeDraft.componentTitle(id)
        }
        return selected.isEmpty ? "Defaults":selected.joined(separator:" · ")
    }
}
private struct TrainInlineRoleEditor: View {
    @StateObject private var draft: TrainInlineRecipeDraft
    @Environment(\.colorScheme) private var colorScheme
    init(session: TrainInlineRecipesSession, role: String) { _draft = StateObject(wrappedValue: TrainInlineRecipeDraft(session: session, role: role)) }
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(TrainInlineRecipeDraft.title(draft.role)).font(.caption.weight(.semibold))
                .foregroundStyle(colorScheme == .dark ? ComponentVisualFamily.training.primary : ComponentVisualFamily.training.secondary)
            FlatChoiceControl(title: TrainInlineRecipeDraft.title(draft.role), selection: draft.componentID ?? "default",
                options: [.init(id: "default", title: "Default")] + TrainInlineRecipeDraft.IDs(draft.role).map { .init(id: $0, title: TrainInlineRecipeDraft.componentTitle($0)) }) {
                    _ = draft.choose($0 == "default" ? nil : $0)
                }.disabled(draft.hasUncommittedNumbers)
                .accessibilityIdentifier("train.inline.\(draft.role).type")
            if draft.componentID != nil {
                if draft.role == "objective" {
                    FlatChoiceControl(title: "Combine objective values", selection: draft.objectiveScalarization, options: [.init(id: "mean", title: "Mean"), .init(id: "sum", title: "Sum")]) { _ = draft.setObjectiveScalarization($0) }
                        .accessibilityIdentifier("train.inline.objective.scalarization")
                }
                if let editor = draft.initialization { InitializationRulesEditorView(context: editor, inlinePresentation: true) }
                else if let editor = draft.gradient { GradientRequestsEditorView(context: editor, inlinePresentation: true) }
                else if let editor = draft.trainability { TrainabilityOwnersEditorView(context: editor, inlinePresentation: true) }
                else {
                    ForEach(draft.fields, id: \.key) { field in
                        if let value = draft.configuration[field.key] {
                            RegistryFieldControl(field: field, value: value, origin: draft.numericOrigin(field.key),
                                currentOrigin: { draft.currentNumericOrigin(field.key) },
                                onDraftChange: { origin, pending in draft.setNumericPending(field.key, origin: origin, pending: pending) },
                                componentID: draft.componentID, configuration: draft.configuration) { value in
                                    draft.setField(field.key, value: value, origin: draft.numericOrigin(field.key))
                                }.accessibilityIdentifier("train.inline.\(draft.role).\(field.key)")
                        }
                    }.id(draft.resetGeneration)
                }
            }
            if let message = draft.rejection ?? ((draft.initialization != nil || draft.gradient != nil || draft.trainability != nil) ? nil:draft.validationMessage) {
                Label(message, systemImage: "exclamationmark.circle").font(.caption).foregroundStyle(MLTheme.rose)
            }
            HStack {
                Button("Apply") { _ = draft.apply() }.disabled(!draft.canApply)
                    .accessibilityIdentifier("train.inline.\(draft.role).apply")
                Button("Revert") { _ = draft.revert() }.disabled(!draft.hasChanges && !draft.hasUncommittedNumbers)
                    .accessibilityIdentifier("train.inline.\(draft.role).revert")
                Spacer(minLength: 0)
                if draft.hasChanges || draft.hasUncommittedNumbers { Text("Unapplied").font(.caption2).foregroundStyle(MLTheme.muted) }
            }.buttonStyle(FlatButtonStyle())
        }.padding(10).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
            .disabled(draft.isBusy).onDisappear { draft.releasePendingEdits() }
    }
}
