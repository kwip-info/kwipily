import Foundation
import KwipMLCore

/// A read-only projection of strictly verified public workflow results, never a lesson receipt.
struct DerivativeResultsPresentation {
    struct Target {
        let name: String
        let shape: [Int]
        let dtype: ContractDType
        let status: DerivativeReportedStatus
        let statusText: String
        let explanation: String
        let tensor: LearningTensor?
    }
    struct Request {
        let id: String
        let output: String
        let phase: String
        let partition: String
        let rowStart: Int
        let rowCount: Int
        let outputShape: [Int]
        let targets: [Target]
    }
    let runID: String
    let device: ContractDevice
    let requests: [Request]
    static let explanation = "A derivative describes the small-change effect: how a selected result responds to a small change in an input or learned setting. Zero is an observed effect; unavailable means no derivative value was returned."
    init(results: DerivativeResults) throws {
        func fail() -> WorkflowPhaseError { WorkflowPhaseError("Verified derivative results cannot be presented with inconsistent context or geometry.") }
        func elements(_ shape: [Int]) throws -> Int {
            guard !shape.isEmpty, shape.count <= 8 else { throw fail() }
            var count = 1
            for d in shape {
                guard d > 0, d <= 65_536, count <= 65_536 / d else { throw fail() }
                count *= d
            }
            return count
        }
        guard !results.runID.isEmpty, results.requests.count <= 32 else { throw fail() }
        var requests: [Request] = [], ids = Set<String>(), appearances = 0
        for result in results.requests {
            guard case .object(let raw) = result.request,
                  let id = raw["id"]?.stringValue, !id.isEmpty, id.utf8.count <= 64, ids.insert(id).inserted,
                  case .object(let endpoint) = raw["output"],
                  let node = endpoint["node"]?.stringValue, !node.isEmpty,
                  let port = endpoint["port"]?.stringValue, !port.isEmpty,
                  let partition = raw["partition"]?.stringValue, ["training", "validation"].contains(partition),
                  case .integer(let start) = raw["row_start"], start >= 0, start <= Int64(Int.max),
                  case .integer(let count) = raw["row_count"], count >= 1, count <= 32,
                  start <= Int64.max - count, result.outputShape.first == Int(count),
                  !result.targets.isEmpty, result.targets.count <= 32 else { throw fail() }
            // Output tensors can exceed the derivative return budget. No output values are copied here.
            guard result.outputShape.count <= 8, result.outputShape.dropFirst().allSatisfy({ $0 > 0 && $0 <= 4096 }) else { throw fail() }
            var names = Set<String>(), targets: [Target] = []
            for target in result.targets {
                guard !target.name.isEmpty, names.insert(target.name).inserted,
                      [.float32, .int64].contains(target.dtype) else { throw fail() }
                let n = try elements(target.shape)
                guard appearances <= 65_536 - n else { throw fail() }
                appearances += n
                let text: String, tensor: LearningTensor?
                switch target.status {
                case .observed:
                    guard target.dtype == .float32, target.reportedCause == nil,
                          let values = target.values, values.count == n,
                          values.allSatisfy({ $0.isFinite && abs($0) <= Double(Float.greatestFiniteMagnitude) && ($0 == 0 || Float($0) != 0) }) else { throw fail() }
                    text = "Measured small-change effects. A zero is an observed value, not a missing result."
                    tensor = .init(shape: target.shape, values: values)
                case .unavailable:
                    guard target.values == nil, let cause = target.reportedCause else { throw fail() }
                    switch cause {
                    case "integerInput": text = "Whole-number input values do not have derivatives. No value was returned."
                    case "frozenParameter": text = "This learned setting was kept fixed. No derivative value was recorded."
                    case "disconnected": text = "This target has no differentiable connection to the selected result. No value was returned."
                    case "sparseGradient": text = "This target produced a sparse derivative that this result table cannot represent. No dense value was returned."
                    default: throw fail()
                    }
                    tensor = nil
                }
                targets.append(.init(name: target.name, shape: target.shape, dtype: target.dtype,
                    status: target.status, statusText: target.status == .observed ? "Observed" : "Unavailable",
                    explanation: text, tensor: tensor))
            }
            requests.append(.init(id: id, output: node + "." + port, phase: "Final snapshot evaluation",
                partition: partition, rowStart: Int(start), rowCount: Int(count), outputShape: result.outputShape, targets: targets))
        }
        self.runID = results.runID; self.device = results.device; self.requests = requests
    }
}
