import Foundation
import Combine
import KwipMLCore

/// Immutable metadata and typed sampled port values from one captured request.
struct WorkflowCapturedDataOutput:Equatable {
    let projectID:String
    let sessionID:UUID
    let address:WorkflowNodeAddress
    let runID:UUID
    let requestIdentity:String
    let capturedRevision:UInt64
    let bindingRevision:UInt64
    let phase:WorkflowExecutionPhase
    let matchesCurrentConstruction:Bool
    let matchesCurrentBindings:Bool
    let ports:[String:JSONValue]
    var dataCompleteness: String? = nil
    var capturedAt: Date? = nil
    /// Completed captures are immutable. Qualify presentation caching by source as well as run.
    var tensorRenderIdentity:String {[projectID,sessionID.uuidString,address.domain.rawValue,address.localID,runID.uuidString,requestIdentity,String(capturedRevision),String(bindingRevision),String(describing:phase)].joined(separator:"\0")}
    /// Pure conversion for the shared legacy renderer. No file or source access.
    var renderedPorts:[String:Any] {ports.mapValues(Self.render)}
    static func render(_ value:JSONValue)->Any {
        switch value {
        case .null:return NSNull()
        case .bool(let value):return value
        case .integer(let value):return NSNumber(value:value)
        case .number(let value):return value
        case .string(let value):return value
        case .array(let values):return values.map(render)
        case .object(let values):return values.mapValues(render)
        }
    }
}

/// Cached-only data/model observations, distinguished by the qualified address.
/// Implementations must reject inactive sessions and
/// resolve qualified addresses against the captured plan, never a local-ID alias.
@MainActor protocol WorkflowDataOutputs:AnyObject {
    var projectID:String {get}
    var sessionID:UUID {get}
    var changes:AnyPublisher<Void,Never> {get}
    func snapshot(for address:WorkflowNodeAddress)->WorkflowCapturedDataOutput?
    func snapshots(for addresses:Set<WorkflowNodeAddress>)->[WorkflowNodeAddress:WorkflowCapturedDataOutput]
    func formulaSampleSession(for consumer:WorkflowNodeAddress,configuration:[String:JSONValue])->TableViewQuerySession?
    func capturedTableViewSession(for endpoint:WorkflowEndpoint) -> TableViewQuerySession?
    func fullTableViewSession(for address: WorkflowNodeAddress) -> TableViewQuerySession?
    func imageArtifactResolver(for address: WorkflowNodeAddress) -> FileImageArtifactResolver?
    func terminalStatus(for address: WorkflowNodeAddress) -> String?
}

extension WorkflowDataOutputs {
    func formulaSampleSession(for consumer:WorkflowNodeAddress,configuration:[String:JSONValue])->TableViewQuerySession? {nil}
    /// Compatibility fallback for providers with only individual cached observations.
    func snapshots(for addresses:Set<WorkflowNodeAddress>)->[WorkflowNodeAddress:WorkflowCapturedDataOutput] {
        addresses.reduce(into:[:]) { result,address in
            if let value=snapshot(for:address),value.address == address,value.projectID == projectID,value.sessionID == sessionID {result[address]=value}
        }
    }
    func capturedTableViewSession(for endpoint:WorkflowEndpoint) -> TableViewQuerySession? { endpoint.port == "table" ? fullTableViewSession(for:endpoint.address) : nil }
    func fullTableViewSession(for address: WorkflowNodeAddress) -> TableViewQuerySession? { nil }
    func imageArtifactResolver(for address: WorkflowNodeAddress) -> FileImageArtifactResolver? { nil }
    func terminalStatus(for address: WorkflowNodeAddress) -> String? { nil }
}
