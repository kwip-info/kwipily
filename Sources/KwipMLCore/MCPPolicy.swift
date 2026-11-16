import Foundation

/// Stable automation actions. Canvas primitives are discovered separately from the
/// current authoring catalog, never duplicated as a second MCP tool registry.
public enum MCPAction:String,Codable,CaseIterable,Hashable,Sendable {
    case catalogList="catalog_list",catalogGet="catalog_get",projectsList="projects_list"
    case projectCreate="project_create",projectOpen="project_open",projectSave="project_save"
    case projectStatus="project_status",projectRename="project_rename",projectClose="project_close"
    case workflowInspect="workflow_inspect",workflowValidate="workflow_validate"
    case patchPreview="patch_preview",patchApply="patch_apply",patchUndo="patch_undo"
    case runStart="run_start",runCancel="run_cancel",jobsList="jobs_list",jobGet="job_get"
    case assetStatusRead="asset_status_read"
    case automationRead="automation_read"
    case schedulePreview="schedule_preview",scheduleApply="schedule_apply"
    case scheduleRunNow="schedule_run_now",scheduleCancel="schedule_cancel"
    case resultsRead="results_read",lessonRead="lesson_read"
    case settingsGet="settings_get",settingsPreview="settings_preview",settingsApply="settings_apply"
    case applicationSettingsGet="application_settings_get",applicationSettingsPreview="application_settings_preview",applicationSettingsApply="application_settings_apply"
    public var requiresProject:Bool {
        ![Self.catalogList,.catalogGet,.projectsList,.projectCreate,.jobsList,.lessonRead,.applicationSettingsGet,.applicationSettingsPreview,.applicationSettingsApply].contains(self)
    }
    public var isReadOnly:Bool {
        [Self.schedulePreview,.assetStatusRead,.automationRead,.projectStatus,.catalogList,.catalogGet,.projectsList,.workflowInspect,.workflowValidate,.patchPreview,.jobsList,.jobGet,.resultsRead,.lessonRead,.settingsGet,.settingsPreview,.applicationSettingsGet,.applicationSettingsPreview].contains(self)
    }
    public var title:String {
        switch self {
        case .assetStatusRead:return "Read retained model asset metadata"
        case .automationRead:return "Read automation metadata"
        case .applicationSettingsGet:return "Read application settings"
        case .applicationSettingsPreview:return "Preview application settings"
        case .applicationSettingsApply:return "Apply application settings"
        case .settingsGet:return "Read project settings"
        case .settingsPreview:return "Preview project settings"
        case .settingsApply:return "Apply project settings"
        default:return rawValue.replacingOccurrences(of:"_",with:" ").capitalized
        }
    }
}
public struct MCPProjectGrant:Codable,Equatable,Sendable {
    /// Creating a managed canvas authorizes the original authoring/run workflow only.
    /// Never derive this from allCases/requiresProject: new privileged actions need
    /// separate project consent. Client and disabled-action checks still intersect.
    public static var createdProject:Self {
        .init(actions:[.projectOpen,.projectSave,.workflowInspect,.workflowValidate,
                       .patchPreview,.patchApply,.patchUndo,.runStart,.runCancel,.jobGet,.resultsRead])
    }
    public var actions:Set<MCPAction>
    public var readCapturedValues:Bool
    public var readHistory:Bool
    public init(actions:Set<MCPAction>=[],readCapturedValues:Bool=false,readHistory:Bool=false){
        self.actions=actions;self.readCapturedValues=readCapturedValues;self.readHistory=readHistory
    }
}
public struct MCPClientGrant:Codable,Equatable,Identifiable,Sendable {
    public var id:String
    public var name:String
    public var actions:Set<MCPAction>
    public var disabledComponents:Set<String>
    public init(id:String,name:String,actions:Set<MCPAction>=[],disabledComponents:Set<String>=[]){
        self.id=id;self.name=name;self.actions=actions;self.disabledComponents=disabledComponents
    }
}
public struct MCPSettings:Codable,Equatable,Sendable {
    public var schemaVersion=1
    public var enabled=false
    /// nil asks the OS for an unused ephemeral port. Never silently substitute a
    /// different port for an explicitly configured value.
    public var fixedPort:Int?=nil
    public var disabledActions:Set<MCPAction>=[]
    public var disabledComponents:Set<String>=["kwip.data.outputDelimited","workflow.saveModel"]
    public var projects:[String:MCPProjectGrant]=[:]
    public var clients:[MCPClientGrant]=[]
    public var auditRetention:Int=200
    public static let factory=MCPSettings()
    public init(){}
    public struct Refusal:Error,LocalizedError,Equatable {
        public let code:String
        public init(code:String){self.code=code}
        public var errorDescription:String? {switch code{
        case "disabled":return "MCP is stopped."
        case "invalid_settings":return "MCP settings are invalid. No access was granted."
        default:return "MCP access is not permitted for this request."
        }}
    }
    public func validated()throws->Self {
        guard schemaVersion==1,fixedPort==nil || (1024...65535).contains(fixedPort!),
              (0...1000).contains(auditRetention),clients.count<=16,projects.count<=512,
              disabledComponents.count<=1024,Set(clients.map(\.id)).count==clients.count else {throw Refusal(code:"invalid_settings")}
        func validIdentifier(_ id:String)->Bool { !id.isEmpty && id.utf8.count<=200 && !id.contains("\0") && !id.contains("/") && !id.contains("\\") }
        guard disabledComponents.allSatisfy(validIdentifier),projects.keys.allSatisfy({UUID(uuidString:$0)?.uuidString==$0}),
              clients.allSatisfy({validIdentifier($0.id) && !$0.name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty && $0.name.utf8.count<=160 && !$0.name.contains("\0") && $0.disabledComponents.count<=1024 && $0.disabledComponents.allSatisfy(validIdentifier)}) else {throw Refusal(code:"invalid_settings")}
        return self
    }
    public func authorize(clientID:String,projectID:String?=nil,action:MCPAction,
                          components:Set<String>=[],requiresValues:Bool=false,requiresHistory:Bool=false)throws {
        _=try validated()
        guard enabled else {throw Refusal(code:"disabled")}
        guard let client=clients.first(where:{$0.id==clientID}),client.actions.contains(action),!disabledActions.contains(action),
              components.isDisjoint(with:disabledComponents),components.isDisjoint(with:client.disabledComponents) else {throw Refusal(code:"not_permitted")}
        if let projectID {
            guard let grant=projects[projectID],grant.actions.contains(action),!requiresValues || grant.readCapturedValues,
                  !requiresHistory || grant.readHistory else {throw Refusal(code:"not_permitted")}
        } else if action.requiresProject || requiresValues || requiresHistory {throw Refusal(code:"not_permitted")}
    }
    public func visibleActions(clientID:String)->[MCPAction] {
        guard enabled,let client=clients.first(where:{$0.id==clientID})else{return []}
        return MCPAction.allCases.filter{client.actions.contains($0) && !disabledActions.contains($0)}
    }
    public func encoded()throws->Data {
        _=try validated();let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        let data=try encoder.encode(self)
        guard data.count<=262144 else {throw Refusal(code:"invalid_settings")};return data
    }
    public static func decode(_ data:Data)throws->Self {
        guard data.count<=262144 else {throw Refusal(code:"invalid_settings")}
        return try JSONDecoder().decode(Self.self,from:data).validated()
    }
}
