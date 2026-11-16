import SwiftUI
import KwipMLCore

struct BackendEstimatorInspector: View {
    @ObservedObject var context: WorkflowControlNodeContext
    var showsInformation=true
    @State private var first = ""
    @State private var second = ""
    @State private var target = ""
    @State private var initialized = false
    private let mappingKey = "estimatorMappingDraft"
    private var mappingChanged: Bool { !DocumentClassificationConfiguration.isDocument(context.configuration) && (context.configuration["featureColumns"] != .array([.string(first), .string(second)]) || context.configuration["targetColumn"] != .string(target)) }
    private var backend: String { context.configuration["backendID"]?.stringValue ?? "sklearn" }
    private var settings: [String: JSONValue] { if case .object(let value) = context.configuration["settings"] { return value }; return [:] }
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if backend=="sklearn" {
                FlatChoiceControl(title:"Estimator task",selection:DocumentClassificationConfiguration.isDocument(context.configuration) ? "document":"numeric",options:[.init(id:"numeric",title:"Numeric binary classifier"),.init(id:"document",title:"Document classification")]) {mode in
                    _=context.authority.configure(mode=="document" ? DocumentClassificationConfiguration.defaults:WorkflowEstimatorConfiguration.sklearnDefaults,expected:context.configuration)
                }
            }
            if DocumentClassificationConfiguration.isDocument(context.configuration) {DocumentClassifierInspector(context:context)} else {
            Text(backend == "sklearn" ? "scikit-learn · Logistic regression · CPU" : "TensorFlow · Dense classifier · CPU").font(.caption.weight(.semibold))
            TextField("First feature column", text: $first).flatField()
            TextField("Second feature column", text: $second).flatField()
            TextField("Target column", text: $target).flatField()
            Button("Apply column mapping") {
                let expected = context.configuration
                var next = expected; next["featureColumns"] = .array([.string(first), .string(second)]); next["targetColumn"] = .string(target)
                do { _ = try WorkflowEstimatorConfiguration.decode(next); _ = context.authority.configure(next, expected: expected); reportMapping() }
                catch { context.authority.message = error.localizedDescription }
            }.buttonStyle(FlatButtonStyle())
            if backend == "sklearn" {
                number("C", title: "Inverse regularization C", defaultValue: 1, minimum: 1e-8, maximum: 1e8)
                number("tolerance", title: "Convergence tolerance", defaultValue: 0.0001, minimum: 1e-12, maximum: 1)
                number("maxIterations", title: "Maximum solver iterations", defaultValue: 200, minimum: 1, maximum: 10000, integer: true)
                Toggle("Fit intercept", isOn: Binding(get: { settings["fitIntercept"] == .bool(true) }, set: { _ = set("fitIntercept", .bool($0)) })).toggleStyle(.checkbox)
            } else {
                ForEach(TensorFlowEstimatorConfiguration.fields, id: \.key) { field in
                    if field.key != "device" {
                        switch field.kind {
                        case .integer(let range): number(field.key, title: field.title, defaultValue: field.defaultValue.doubleValue ?? 0, minimum: Double(range.lowerBound), maximum: Double(range.upperBound), integer: true)
                        case .positiveDecimal(let maximum),.nonnegativeDecimal(let maximum): number(field.key, title: field.title, defaultValue: field.defaultValue.doubleValue ?? 0, minimum: 0, maximum: maximum)
                        case .choice: EmptyView()
                        }
                    }
                }
            }
            }
            if showsInformation {
                InspectorInformationSection {Text(DocumentClassificationConfiguration.isDocument(context.configuration) ? "Single-label document classification uses frozen Training-only TF-IDF and sparse L2/lbfgs logistic regression. Fit evaluates Validation; prediction applies the saved processor without fitting. Save model packages vocabulary/IDF, exact class order and coefficients; Load model imports only inert state. No calibration, unknown-class detection, OCR or optimizer continuation is implied.":Self.information(backend:backend))}
            }
        }.task(id: context.contextIdentity) {
            initialized = false
            if case .array(let columns) = context.configuration["featureColumns"], columns.count == 2 { first = columns[0].stringValue ?? ""; second = columns[1].stringValue ?? "" }
            target = context.configuration["targetColumn"]?.stringValue ?? ""
            initialized = true; reportMapping()
        }.onChange(of: [first, second, target]) { _, _ in reportMapping() }
            .onDisappear { context.setNumericPending(origin: context.numericOrigin(key: mappingKey), key: mappingKey, pending: false) }
    }
    static func information(backend:String)->String {
        return backend == "sklearn" ? "Fits z = Xw + b on training rows, then maps scores through sigmoid. Validation does not update coefficients. Smaller C strengthens L2 regularization. The full-batch solver has no minibatch or initialization seed; change partition seeds on the Split component." : "Constructs the connected dense model in Keras. Training uses the configured seed, epochs, batch size and learning rate. Prepared training and validation remain separate; no gradients cross into preprocessing."
    }
    private func reportMapping() {
        guard initialized else { return }
        context.setNumericPending(origin: context.numericOrigin(key: mappingKey), key: mappingKey, pending: mappingChanged)
    }
    @discardableResult private func set(_ key: String, _ value: JSONValue) -> Bool {
        guard !mappingChanged else { context.authority.message = "Apply the column mapping before changing estimator settings."; return false }
        let expected = context.configuration
        var next = expected, adjusted = settings; adjusted[key] = value; next["settings"] = .object(adjusted)
        do {
            if backend == "sklearn" { try SklearnEstimatorConfiguration.validate(adjusted) } else { try TensorFlowEstimatorConfiguration.validate(adjusted) }
            return context.authority.configure(next, expected: expected)
        } catch { context.authority.message = error.localizedDescription; return false }
    }
    private func number(_ key: String, title: String, defaultValue: Double, minimum: Double, maximum: Double, integer: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption)
            BoundedNumberField(title: title, value: integer ? .wholeDisplay(settings[key]?.doubleValue ?? defaultValue) : .decimal(settings[key]?.doubleValue ?? defaultValue),
                domain: integer ? .integer(minimum: minimum, maximum: maximum) : .decimal(minimum: minimum, maximum: maximum),
                origin: context.numericOrigin(key: key), currentOrigin: { context.currentNumericOrigin(key: key) },
                onDraftChange: { origin, pending in context.setNumericPending(origin: origin, key: key, pending: pending) }) { value in
                    switch value { case .integer(let number): return set(key, .integer(number)); case .decimal(let number): return set(key, .number(number)) }
                }
        }
    }
}
