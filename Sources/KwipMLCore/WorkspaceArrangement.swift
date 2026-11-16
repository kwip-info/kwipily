import Foundation

public struct CanvasViewState:Codable,Equatable {
    public var zoom:Double;public var scrollX:Double;public var scrollY:Double
    public var selectedNodeIDs:Set<String>;public var selectedEdgeIDs:Set<String>
    public var overviewVisible:Bool
    public var presentation:CanvasPresentationSettings
    public init(zoom:Double=0.65,scrollX:Double=0,scrollY:Double=0,selectedNodeIDs:Set<String>=[],selectedEdgeIDs:Set<String>=[],presentation:CanvasPresentationSettings=CanvasPresentationSettings(),overviewVisible:Bool=true){self.zoom=zoom;self.scrollX=scrollX;self.scrollY=scrollY;self.selectedNodeIDs=selectedNodeIDs;self.selectedEdgeIDs=selectedEdgeIDs;self.presentation=presentation;self.overviewVisible=overviewVisible}
    private enum CodingKeys:String,CodingKey {case zoom,scrollX,scrollY,selectedNodeIDs,selectedEdgeIDs,presentation,overviewVisible}
    public init(from decoder:Decoder)throws {
        let values=try decoder.container(keyedBy:CodingKeys.self)
        zoom=try values.decode(Double.self,forKey:.zoom);scrollX=try values.decode(Double.self,forKey:.scrollX);scrollY=try values.decode(Double.self,forKey:.scrollY)
        selectedNodeIDs=try values.decode(Set<String>.self,forKey:.selectedNodeIDs);selectedEdgeIDs=try values.decode(Set<String>.self,forKey:.selectedEdgeIDs)
        overviewVisible=try values.decodeIfPresent(Bool.self,forKey:.overviewVisible) ?? true
        presentation=values.contains(.presentation) ? try values.decode(CanvasPresentationSettings.self,forKey:.presentation):CanvasPresentationSettings()
    }
}
public struct WorkspaceViewState:Codable,Equatable {
    public var showLibrary:Bool;public var showInspector:Bool;public var showPreview:Bool
    public var activeSection:String;public var modelCanvas:CanvasViewState;public var dataCanvas:CanvasViewState
    public var compositeCanvasStates:[String:CanvasViewState]
    public var blockCanvasStates:[String:CanvasViewState]
    public var compactNavigation:Bool?
    public var panelLayouts:[String:PanelLayoutPreference]
    public init(showLibrary:Bool=true,showInspector:Bool=true,showPreview:Bool=true,activeSection:String="Build",modelCanvas:CanvasViewState=CanvasViewState(),dataCanvas:CanvasViewState=CanvasViewState(),panelLayouts:[String:PanelLayoutPreference]=[:],compactNavigation:Bool?=nil,compositeCanvasStates:[String:CanvasViewState]=[:],blockCanvasStates:[String:CanvasViewState]=[:]){self.showLibrary=showLibrary;self.showInspector=showInspector;self.showPreview=showPreview;self.activeSection=activeSection;self.modelCanvas=modelCanvas;self.dataCanvas=dataCanvas;self.panelLayouts=panelLayouts;self.compactNavigation=compactNavigation;self.compositeCanvasStates=compositeCanvasStates;self.blockCanvasStates=blockCanvasStates}
    private enum CodingKeys:String,CodingKey {case showLibrary,showInspector,showPreview,activeSection,modelCanvas,dataCanvas,panelLayouts,compactNavigation,compositeCanvasStates,blockCanvasStates}
    public init(from decoder:Decoder)throws {
        let values=try decoder.container(keyedBy:CodingKeys.self)
        showLibrary=try values.decode(Bool.self,forKey:.showLibrary);showInspector=try values.decode(Bool.self,forKey:.showInspector);showPreview=try values.decode(Bool.self,forKey:.showPreview)
        activeSection=try values.decode(String.self,forKey:.activeSection)
        modelCanvas=try values.decode(CanvasViewState.self,forKey:.modelCanvas);dataCanvas=try values.decode(CanvasViewState.self,forKey:.dataCanvas)
        compositeCanvasStates=values.contains(.compositeCanvasStates) ? try values.decode([String:CanvasViewState].self,forKey:.compositeCanvasStates):[:]
        blockCanvasStates=try values.decodeIfPresent([String:CanvasViewState].self,forKey:.blockCanvasStates) ?? [:]
        compactNavigation=try values.decodeIfPresent(Bool.self,forKey:.compactNavigation)
        panelLayouts=try values.decodeIfPresent([String:PanelLayoutPreference].self,forKey:.panelLayouts) ?? [:]
    }
    public func validate()throws {
        guard blockCanvasStates.count<=128,blockCanvasStates.keys.allSatisfy(ModelChildCanvasKey.isValid) else {throw WorkspaceArrangementError("Invalid or excessive Block instance canvas contexts (maximum 128).")}
        guard compositeCanvasStates.count<=65,compositeCanvasStates.keys.allSatisfy(Self.validCompositeCanvasKey) else {throw WorkspaceArrangementError("Invalid or excessive composite canvas contexts (maximum 65).")}
        guard panelLayouts.count<=WorkspacePanelScope.allCases.count else {throw WorkspaceArrangementError("Too many panel layouts.")}
        for (key,value) in panelLayouts {guard let scope=WorkspacePanelScope(rawValue:key) else {throw WorkspaceArrangementError("Unknown panel layout.")};try value.validate(scope:scope)}
        guard ["Projects","Data","Build","Run","Code","Examples","Experiments","Settings","Learn","History"].contains(activeSection) else {throw WorkspaceArrangementError("Unknown workspace section.")}
        for state in [modelCanvas,dataCanvas]+Array(compositeCanvasStates.values)+Array(blockCanvasStates.values) {guard state.zoom.isFinite,CanvasZoomLimits.supported.contains(state.zoom),state.scrollX.isFinite,state.scrollY.isFinite,(-1_000_000...1_000_000).contains(state.scrollX),(-1_000_000...1_000_000).contains(state.scrollY),state.selectedNodeIDs.count<=256,state.selectedEdgeIDs.count<=512,(state.selectedNodeIDs.union(state.selectedEdgeIDs)).allSatisfy({!$0.isEmpty && $0.utf8.count<=16384}) else {throw WorkspaceArrangementError("Invalid canvas view state.")}}
    }
    public static func validCompositeCanvasKey(_ key:String)->Bool {
        if key == "root" { return true }
        guard key.count<=1024,key.hasPrefix("composite:"),let separator=key.lastIndex(of:"@") else {return false}
        let identity=key[key.index(key.startIndex,offsetBy:10)..<separator]
        let pieces=key[key.index(after:separator)...].split(separator:".",omittingEmptySubsequences:false)
        return !identity.isEmpty && pieces.count==3 && pieces.allSatisfy { part in
            guard let value=Int(part),value>=0 else {return false};return String(value)==part
        }
    }
}
public struct WorkspaceArrangement:Codable,Equatable,Identifiable {
    public var schemaVersion:Int;public let id:String;public var name:String;public var state:WorkspaceViewState
    public init(schemaVersion:Int=1,id:String=UUID().uuidString,name:String,state:WorkspaceViewState){self.schemaVersion=schemaVersion;self.id=id;self.name=name;self.state=state}
}
public struct WorkspaceArrangementError:Error,LocalizedError {
    public let message:String;public var errorDescription:String? {message}
    public init(_ message:String){self.message=message}
}
/// Local UI sidecars only: no graph configuration, data rows, paths or access grants.
public final class WorkspaceArrangementLibrary {
    public let directory:URL
    public init(directory:URL){self.directory=directory}
    public func list()throws->[WorkspaceArrangement] {
        let dir=directory.appendingPathComponent("named")
        guard FileManager.default.fileExists(atPath:dir.path) else{return []}
        try checkPath(dir)
        let paths=try FileManager.default.contentsOfDirectory(at:dir,includingPropertiesForKeys:nil).filter{$0.pathExtension=="json"}
        guard paths.count<=100 else {throw WorkspaceArrangementError("At most 100 named arrangements are supported.")}
        return try paths.map {path in let record:WorkspaceArrangement=try read(path);try validate(record);guard path.deletingPathExtension().lastPathComponent==record.id else {throw WorkspaceArrangementError("Arrangement identity mismatch.")};return record}.sorted{$0.name.localizedStandardCompare($1.name) == .orderedAscending}
    }
    @discardableResult public func save(name:String,state:WorkspaceViewState,id:String?=nil)throws->WorkspaceArrangement {
        let record=WorkspaceArrangement(id:id ?? UUID().uuidString,name:name.trimmingCharacters(in:.whitespacesAndNewlines),state:state);try validate(record)
        if id==nil {guard try list().count<100 else {throw WorkspaceArrangementError("At most 100 named arrangements are supported.")}}
        try write(record,to:path(record.id,folder:"named"));return record
    }
    @discardableResult public func rename(id:String,name:String)throws->WorkspaceArrangement {let record:WorkspaceArrangement=try read(path(id,folder:"named"));try validate(record);return try save(name:name,state:record.state,id:id)}
    public func delete(id:String)throws {let url=try path(id,folder:"named");try checkPath(url);try FileManager.default.removeItem(at:url)}
    public func saveProjectState(_ state:WorkspaceViewState,projectID:String)throws {try state.validate();try write(WorkspaceArrangement(id:projectID,name:"Project workspace",state:state),to:path(projectID,folder:"projects"))}
    public func projectState(projectID:String)throws->WorkspaceViewState? {let url=try path(projectID,folder:"projects");guard FileManager.default.fileExists(atPath:url.path) else{return nil};let record:WorkspaceArrangement=try read(url);try validate(record);guard record.id==projectID else {throw WorkspaceArrangementError("Project view identity mismatch.")};return record.state}
    private func validate(_ record:WorkspaceArrangement)throws {guard record.schemaVersion==1,UUID(uuidString:record.id) != nil,!record.name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,record.name.count<=80 else {throw WorkspaceArrangementError("Invalid arrangement schema, name or identity.")};try record.state.validate()}
    private func path(_ id:String,folder:String)throws->URL {guard UUID(uuidString:id) != nil else {throw WorkspaceArrangementError("Workspace identity must be a UUID.")};return directory.appendingPathComponent(folder).appendingPathComponent(id+".json")}
    private func checkPath(_ url:URL)throws {for candidate in [directory,url.deletingLastPathComponent(),url] {if (try? candidate.resourceValues(forKeys:[.isSymbolicLinkKey]).isSymbolicLink)==true {throw WorkspaceArrangementError("Workspace sidecars cannot use symbolic links.")}}}
    private func read<T:Decodable>(_ url:URL)throws->T {try checkPath(url);let values=try url.resourceValues(forKeys:[.fileSizeKey,.isRegularFileKey]);guard values.isRegularFile==true,(values.fileSize ?? Int.max)<=1_000_000 else {throw WorkspaceArrangementError("Workspace sidecar exceeds bounds.")};return try JSONDecoder().decode(T.self,from:Data(contentsOf:url))}
    private func write<T:Encodable>(_ value:T,to url:URL)throws {try checkPath(url);try FileManager.default.createDirectory(at:url.deletingLastPathComponent(),withIntermediateDirectories:true);let data=try JSONEncoder().encode(value);guard data.count<=1_000_000 else {throw WorkspaceArrangementError("Workspace sidecar exceeds bounds.")};try data.write(to:url,options:.atomic)}
}
