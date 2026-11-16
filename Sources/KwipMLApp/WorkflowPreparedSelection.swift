import Foundation
import KwipMLCore

/// One verified preparation result, with an explicit mapping variant.
enum WorkflowPreparedSelection: Equatable {
    case legacy(PreparedTrainingSelection)
    case objective(PreparedObjectiveSelection)
    var reference: PreparedDataReference {
        switch self { case .legacy(let value): return value.reference; case .objective(let value): return value.reference }
    }
    var manifestPath: String { reference.manifestPath }
    var manifestSHA256: String { reference.manifestSHA256 }
    var graphDigest: String { reference.graphDigest }
    var trainingNode: String { reference.trainingNode }
    var trainingPort: String { reference.trainingPort }
    var validationNode: String { reference.validationNode }
    var validationPort: String { reference.validationPort }
    func validate() throws {
        switch self { case .legacy(let value): try value.validate(); case .objective(let value): try value.validate() }
    }
}
