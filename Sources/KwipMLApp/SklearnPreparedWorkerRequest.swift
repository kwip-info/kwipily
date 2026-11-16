import Foundation
import KwipMLCore

struct SklearnPreparedWorkerRequest: Codable, Equatable {
    var schemaVersion = 1
    var backendID = "sklearn"
    var capturePredictions: Bool? = nil
    struct Model:Codable,Equatable {let path:String;let sha256:String;var preparationPath:String?=nil;var preparationSHA256:String?=nil}
    var model:Model? = nil
    let configuration: [String: JSONValue]
    let prepared: Prepared
    struct Prepared: Codable, Equatable {
        let manifest: String
        let manifestSHA256: String
        let graphDigest: String
        let trainingNode: String
        let trainingPort: String
        let validationNode: String
        let validationPort: String
        let featureColumns: [String]
        let targetColumn: String
        var selection: PreparedTrainingSelection {
            .init(manifestPath: manifest, manifestSHA256: manifestSHA256, graphDigest: graphDigest,
                  trainingNode: trainingNode, trainingPort: trainingPort, validationNode: validationNode, validationPort: validationPort,
                  featureColumns: featureColumns, targetColumn: targetColumn)
        }
    }
    init(configuration: [String: JSONValue], selection: PreparedTrainingSelection,model:Model?=nil) throws {
        self.model=model
        self.configuration = configuration
        self.prepared = .init(manifest: selection.manifestPath, manifestSHA256: selection.manifestSHA256,
            graphDigest: selection.graphDigest, trainingNode: selection.trainingNode, trainingPort: selection.trainingPort,
            validationNode: selection.validationNode, validationPort: selection.validationPort,
            featureColumns: selection.featureColumns, targetColumn: selection.targetColumn)
        try validate()
    }
    func validate() throws {
        guard schemaVersion == 1, backendID == "sklearn" else { throw WorkflowArchiveError("Unsupported sklearn request.") }
        let documentTask=DocumentClassificationConfiguration.isDocument(configuration)
        if documentTask {
            guard Set(configuration.keys)==Set(["task","settings"]+(configuration["operation"] != nil ? ["operation"]:[])+(configuration["reviewThreshold"] != nil ? ["reviewThreshold"]:[])),case .object(let settings)=configuration["settings"] else {throw WorkflowArchiveError("Invalid document classifier request.")}
            try DocumentClassificationConfiguration.validateEstimator(settings)
        } else if PrebuiltModelConfiguration.isPrebuilt(configuration) {
            guard let a=configuration["algorithm"]?.stringValue,case .object(let settings)=configuration["settings"],Set(configuration.keys)==["task","algorithm","operation","settings"] else {throw WorkflowArchiveError("Invalid Prebuilt models request.")}
            try PrebuiltModelConfiguration.validateSettings(algorithm:a,settings:settings)
        } else {try SklearnEstimatorConfiguration.validate(configuration)}
        if PrebuiltModelConfiguration.isPrebuilt(configuration) {try prepared.selection.reference.validate()} else {try prepared.selection.validate()}
        if configuration["operation"] == .string("predict") || configuration["operation"] == .string("evaluate") {
            guard let model,model.path.hasSuffix("/input/model-state.json"),WorkflowArchiveCodec.isHash(model.sha256) else {throw WorkflowArchiveError("Prediction needs a captured model package state.")}
            if model.preparationPath != nil || model.preparationSHA256 != nil {
                guard PrebuiltModelConfiguration.isPrebuilt(configuration),let path=model.preparationPath,path.hasSuffix("/input/model-preparation.json"),let pin=model.preparationSHA256,WorkflowArchiveCodec.isHash(pin) else {throw WorkflowArchiveError("Capture the model-bound learned preparation with its model.")}
            }
        } else if model != nil {throw WorkflowArchiveError("Fitting cannot import replacement classifier state.")}

        guard prepared.manifest.utf8.count <= 4096, (PrebuiltModelConfiguration.isPrebuilt(configuration) ? (1...128).contains(prepared.featureColumns.count):prepared.featureColumns.count == (documentTask ? 1:2)),
              WorkflowArchiveCodec.isHash(prepared.manifestSHA256), WorkflowArchiveCodec.isHash(prepared.graphDigest) else { throw WorkflowFileLeaseError.unsafePath }
    }
    func encoded() throws -> Data { try validate(); return try WorkflowArchiveCodec.encode(self) }
    static func decode(_ data: Data) throws -> Self {
        guard data.count <= 100_000, let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              Set(object.keys).isSuperset(of: ["schemaVersion", "backendID", "configuration", "prepared"]),
              Set(object.keys).isSubset(of: ["schemaVersion", "backendID", "configuration", "prepared", "capturePredictions", "model"]),
              let prepared = object["prepared"] as? [String: Any], Set(prepared.keys) == ["manifest", "manifestSHA256", "graphDigest", "trainingNode", "trainingPort", "validationNode", "validationPort", "featureColumns", "targetColumn"] else { throw WorkflowFileLeaseError.unsafePath }
        let result = try JSONDecoder().decode(Self.self, from: data); try result.validate(); return result
    }
}
