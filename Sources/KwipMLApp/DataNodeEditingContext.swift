import SwiftUI
import KwipMLCore

/// A single node's editing authority. Implementations project their canonical owner;
/// neither the shared inspector nor its widgets own a mutable graph or file grant.
@MainActor protocol DataNodeEditing: ObservableObject {
    var decisionAppearance:AppAppearanceState {get}
    var hasPendingDecisionDraft:Bool {get}
    var connectedLearnedPreparation:[String:JSONValue]? {get}
    var connectedDecisionQuestions:[String:JSONValue]? {get}
    var decisionQuestionResources:[(String,[String:JSONValue])] {get}
    func makeGameRuleValidationSession()->GameRuleValidationSession?
    var databaseOutputs:DatabaseOutputStore? {get}
    var apiFetchReceipt:[String:JSONValue]? {get}
    var contextIdentity: String { get }
    var node: DataFlowNode? { get }
    var authoredName: String { get }
    var operatorTitle: String { get }
    var operatorDefinition: DataFlowOperatorDefinition? { get }
    var isCurrent: Bool { get }
    var isBusy: Bool { get }
    var statusMessage: String { get }
    var upstreamSchema: [DataColumnSchema] { get }
    func inputSourceName(forInput port:String)->String?
    func upstreamSchema(forInput port:String)->[DataColumnSchema]
    var schemaDisclosure: String { get }
    var usesFixtures: Bool { get }
    var fixtureColumns: [DataColumnSchema] { get }
    var sourceDisplayName: String? { get }
    var sourceDatabaseProjectID: String? { get }
    @discardableResult func connectDatabaseSource() -> Bool
    var sourceIsLinked: Bool { get }
    var sourceInspectionPreview: SourceInspectionResult? { get }
    var sourceInspectionStatus: SourceInspectionStatus? { get }
    var sourcePermissionSummary: String { get }
    var outputDestinationSummary: String { get }
    @discardableResult func chooseOutputDestination() -> Bool
    var sourceSeparatorValues: [String] { get }
    var outputs: [String: Any]? { get }
    var fullTableViewSession: TableViewQuerySession? { get }
    func formulaSampleSession(configuration:[String:JSONValue]) -> TableViewQuerySession?
    func capturedTableViewSession(port:String) -> TableViewQuerySession?
    var imageArtifactResolver: FileImageArtifactResolver? { get }
    var outputPopulationLabel: String? { get }
    var outputCapturedAt: Date? { get }
    var sourcePreviewCompletedAt: Date? { get }
    var outputStatus: String { get }
    var outputIsCurrent: Bool { get }
    var retainedRows: String? { get }
    var outputDisclosure:String? {get}
    var emptyOutputMessage:String {get}
    @discardableResult func configure(_ config: [String: JSONValue], expected: [String: JSONValue]) -> Bool
    @discardableResult func applyTableConfiguration(_ config: [String: JSONValue], expected: [String: JSONValue]) -> Bool
    @discardableResult func rename(_ name: String) -> Bool
    func numericOrigin(key: String) -> String
    func currentNumericOrigin(key: String) -> String?
    func setNumericPending(origin: String, key: String, pending: Bool)
    @discardableResult func chooseSource() -> Bool
    @discardableResult func refreshSource() -> Bool
    @discardableResult func setSeparator(_ value: String, expected: [String: JSONValue]) -> Bool
}

extension DataNodeEditing {
    var decisionAppearance:AppAppearanceState {AppAppearanceState()}
    var hasPendingDecisionDraft:Bool {false}
    var connectedDecisionQuestions:[String:JSONValue]? {nil}
    var connectedLearnedPreparation:[String:JSONValue]? {nil}
    var decisionQuestionResources:[(String,[String:JSONValue])] {[]}
    func makeGameRuleValidationSession()->GameRuleValidationSession? {nil}
    var apiFetchReceipt:[String:JSONValue]? {nil}
    var databaseOutputs:DatabaseOutputStore? {nil}
    func formulaSampleSession(configuration:[String:JSONValue]) -> TableViewQuerySession? {nil}
    var fullTableViewSession: TableViewQuerySession? { nil }
    func capturedTableViewSession(port:String) -> TableViewQuerySession? {port == "table" ? fullTableViewSession : nil}
    var imageArtifactResolver: FileImageArtifactResolver? { nil }
    var outputPopulationLabel: String? { nil }
    var outputCapturedAt: Date? { nil }
    var sourcePreviewCompletedAt: Date? { nil }
    var sourceDatabaseProjectID: String? { nil }
    @discardableResult func connectDatabaseSource() -> Bool { false }
    var outputIsCurrent: Bool { false }
    var operatorDefinition: DataFlowOperatorDefinition? { nil }
    var sourceInspectionPreview: SourceInspectionResult? { nil }
    var sourceInspectionStatus: SourceInspectionStatus? { nil }
    var outputDestinationSummary: String { "Output destination selection is unavailable in this editor." }
    @discardableResult func chooseOutputDestination() -> Bool { false }
    var sourcePermissionSummary: String { sourceIsLinked ? "Local source linked. Refresh to verify access and inspect its current contents." : "No local source permission selected." }
    func inputSourceName(forInput port:String)->String? { nil }
    func upstreamSchema(forInput port:String)->[DataColumnSchema] { [] }
    var outputDisclosure:String? {nil}
    var emptyOutputMessage:String {usesFixtures ? "Lesson assessment inspects measured behavior separately. This editor shows declared connections and configuration." : "Prepare data to inspect actual values and lineage here."}
    var sourceColumns: [String] { upstreamSchema.map(\.name) }
}

enum DataNodeSourceOperation { case choose, refresh, separator(String) }
