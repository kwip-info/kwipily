import Foundation
import Combine
import KwipMLCore

/// Private session service. Read methods return cached presentation only: no grants,
/// prompts, hashing, snapshots, filesystem reads or worker launches during rendering.
/// Implementations must independently validate the owner lease before consequential
/// actions and again after a file picker returns. They do not own workflow configuration.
@MainActor protocol WorkflowDataSources: AnyObject {
    var databaseOutputs:DatabaseOutputStore? {get}
    func apiFetchReceipt(for address:WorkflowNodeAddress)->[String:JSONValue]?
    var projectID: String { get }
    var sessionID: UUID { get }
    var isBusy: Bool { get }
    var statusMessage: String { get }
    var changes: AnyPublisher<Void,Never> { get }
    func recordedSchema(for address: WorkflowNodeAddress, in document: WorkflowDocument) -> [DataColumnSchema]
    func recordedSchema(for address: WorkflowNodeAddress, in document: WorkflowDocument, inputPort:String) -> [DataColumnSchema]
    func outputDestinationSummary(for address: WorkflowNodeAddress) -> String
    @discardableResult func chooseOutputDestination(for address: WorkflowNodeAddress, lease: WorkflowEditorLease) -> Bool
    @discardableResult func chooseOutputDestinations(for addresses:Set<WorkflowNodeAddress>, lease:WorkflowEditorLease) -> Bool
    func sourceInspectionCompletedAt(for address: WorkflowNodeAddress) -> Date?
    func sourceInspectionPreview(for address: WorkflowNodeAddress) -> SourceInspectionResult?
    func sourceInspectionStatus(for address: WorkflowNodeAddress) -> SourceInspectionStatus?
    func bindingDisplayName(for address: WorkflowNodeAddress) -> String?
    @discardableResult func connectDatabaseSource(for address: WorkflowNodeAddress, lease: WorkflowEditorLease) -> Bool
    @discardableResult func chooseSource(for address: WorkflowNodeAddress, lease: WorkflowEditorLease) -> Bool
    @discardableResult func inspectSource(for address: WorkflowNodeAddress, lease: WorkflowEditorLease) -> Bool
}

extension WorkflowDataSources {
    @discardableResult func chooseOutputDestinations(for addresses:Set<WorkflowNodeAddress>,lease:WorkflowEditorLease)->Bool {
        guard addresses.count == 1,let address=addresses.first else {return false}
        return chooseOutputDestination(for:address,lease:lease)
    }
    func apiFetchReceipt(for address:WorkflowNodeAddress)->[String:JSONValue]? {sourceInspectionPreview(for:address)?.apiFetch}
    var databaseOutputs:DatabaseOutputStore? {nil}
    func sourceInspectionCompletedAt(for address: WorkflowNodeAddress) -> Date? { nil }
    @discardableResult func connectDatabaseSource(for address: WorkflowNodeAddress, lease: WorkflowEditorLease) -> Bool { false }
    func outputDestinationSummary(for address: WorkflowNodeAddress) -> String { "Output destination selection is unavailable in this editor." }
    @discardableResult func chooseOutputDestination(for address: WorkflowNodeAddress, lease: WorkflowEditorLease) -> Bool { false }
    func sourceInspectionPreview(for address: WorkflowNodeAddress) -> SourceInspectionResult? { nil }
    func sourceInspectionStatus(for address: WorkflowNodeAddress) -> SourceInspectionStatus? { nil }
    func recordedSchema(for address:WorkflowNodeAddress,in document:WorkflowDocument,inputPort:String)->[DataColumnSchema] { [] }
}
