import Foundation
import KwipMLCore

struct SklearnRuntimeCapabilities: Equatable {
    static let expected: JSONValue = .object([
        "schemaVersion": .integer(1), "backendID": .string("sklearn"), "runtimeVersion": .string("1.7.2"),
        "prebuiltCatalogSHA256":.string(PrebuiltModelConfiguration.catalogSHA256),
        "devices": .array([.string("cpu")]),
        "supportedOperations": .array(["fit", "predict", "evaluate", "save", "load", "export"].map(JSONValue.string)),
        "unsupportedOperations": .array(["partialFit", "resume", "finetune", "autograd", "gpu"].map(JSONValue.string))
    ])
    static func decode(_ data: Data) throws -> Self {
        guard data.count <= 4096, try JSONDecoder().decode(JSONValue.self, from: data) == expected else { throw DataFlowContractError("Unsupported sklearn runtime capability contract.") }
        return .init()
    }
}
