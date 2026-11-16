import Foundation
import Darwin
import SwiftUI
import KwipMLCore

struct GameEpisodeRecord:Codable,Identifiable {
    var schemaVersion=1
    let id:String
    let address:WorkflowNodeAddress
    let document:WorkflowDocument
    let files:[WorkflowArchiveFile]
    let finishedAt:Date
    let validateOnly:Bool
    func directory(_ root:URL)throws->URL {
        guard let pointer=realpath(root.path,nil) else {throw WorkflowFileLeaseError.unsafePath}
        defer {free(pointer)}
        return URL(fileURLWithPath:String(cString:pointer)).appendingPathComponent("workflow-operations/\(document.projectID)/\(id)")
    }
    func verify(root:URL)throws->[String:TableViewQuerySession.Artifact] {
        guard schemaVersion==1,UUID(uuidString:id)?.uuidString==id,!validateOnly,files.count==4,Set(files.map(\.path))==["artifacts/session.json","artifacts/summary.table.json","artifacts/episodes.table.json","artifacts/moves.table.json"],files.reduce(0,{$0+$1.bytes})<=8_000_000 else {throw WorkflowFileLeaseError.unsafePath}
        let directory=try directory(root),request=try JSONDecoder().decode(WorkflowToolRequest.self,from:DiagnosticFileIO.read(root:directory,parts:["request.json"],limit:262_144))
        guard request.operationID==id,request.projectID==document.projectID,request.toolMode == .gameSession else {throw WorkflowFileLeaseError.unsafePath}
        do {try request.validate(directory:directory,verifyInputs:true)} catch {throw WorkflowArchiveError("Recorded game request verification failed: \(error.localizedDescription)")}
        let inner=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:directory,parts:["input","session.json"],limit:2_000_000))
        guard inner["validateOnly"] == .bool(false),let node=document.nodes.first(where:{$0.address==address}),case .gameSession(let config)=node.payload,inner["configuration"] == .object(try GameSessionConfiguration(config).values) else {throw WorkflowArchiveError("Recorded session settings differ from their captured request.")}
        let rules=document.edges.filter{$0.to.address==address && $0.to.port=="definition"}
        guard rules.count==1,let source=rules.first,let definition=document.nodes.first(where:{$0.address==source.from.address}),case .dataOperator(GameRuleConfiguration.operatorID,let values)=definition.payload,inner["definition"]==values["definition"] else {throw WorkflowArchiveError("Recorded rules differ from their captured request.")}
        var result:[String:TableViewQuerySession.Artifact]=[:]
        for f in files {do {try WorkflowPreviewToolContract.verifyFile(file:f,directory:directory,limit:8_000_000)} catch {throw WorkflowArchiveError("Recorded game artifact \(f.path) verification failed: \(error.localizedDescription)")}}
        for port in ["summary","episodes","moves"] {guard let file=files.first(where:{$0.path=="artifacts/"+port+".table.json"})else{throw WorkflowFileLeaseError.unsafePath};result[port] = .init(indexURL:directory.appendingPathComponent(file.path),indexSHA256:file.sha256,sourceSHA256:file.sha256,inputKind:"table")}
        return result
    }
    static func list(root:URL,projectID:String)throws->[Self] {
        guard UUID(uuidString:projectID)?.uuidString==projectID else {throw WorkflowFileLeaseError.unsafePath}
        let directory=root.appendingPathComponent("workflow-operations/\(projectID)")
        guard FileManager.default.fileExists(atPath:directory.path)else{return []}
        let candidates=try FileManager.default.contentsOfDirectory(at:directory,includingPropertiesForKeys:[.isDirectoryKey,.isSymbolicLinkKey])
        guard candidates.count<=20_000 else{throw WorkflowArchiveError("Too many local operations to scan.")}
        var records:[Self]=[]
        for entry in candidates where UUID(uuidString:entry.lastPathComponent)?.uuidString==entry.lastPathComponent {
            try Task.checkCancellation()
            guard let resource=try? entry.resourceValues(forKeys:[.isDirectoryKey,.isSymbolicLinkKey]),resource.isDirectory==true,resource.isSymbolicLink != true,let raw=try? DiagnosticFileIO.read(root:entry,parts:["game-session-record.json"],limit:2_000_000),let record=try? JSONDecoder().decode(Self.self,from:raw),record.id==entry.lastPathComponent,record.document.projectID==projectID,!record.validateOnly else{continue}
            records.append(record)
        }
        return Array(records.sorted{$0.finishedAt>$1.finishedAt}.prefix(100))
    }
}
struct GameEpisodeHistoryView:View {
    let record:GameEpisodeRecord
    @ObservedObject var workspace:WorkspaceStore
    let isActive:()->Bool
    @State private var port="summary"
    @State private var artifacts:[String:TableViewQuerySession.Artifact]=[:]
    @State private var query:TableViewQuerySession?
    @State private var message="Verifying recorded game session…"
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            GameSessionOutputChoices(port:$port)
            Text("Frozen-policy inference · exact recorded rules and snapshots. No live canvas or original source is substituted.").font(.caption).foregroundStyle(MLTheme.muted)
            if let query {TableViewOutputView(session:query,executionBusy:workspace.isBusy)}
            if !message.isEmpty {Text(message).font(.caption).foregroundStyle(MLTheme.muted)}
        }.task(id:record.id){do{let root=workspace.workspaceDirectory;let values=try await WorkflowHistoryVerificationWork.run{try record.verify(root:root)};guard !Task.isCancelled,isActive()else{return};artifacts=values;message="";select()}catch{message=error.localizedDescription}}
        .onChange(of:port){_,_ in select()}.onDisappear{query?.cancelIfUnowned()}
    }
    private func select(){query?.cancelIfUnowned();guard let artifact=artifacts[port]else{query=nil;return};query=workspace.makeTableViewQuerySession(artifact:artifact,projectID:record.document.projectID,accepts:isActive)}
}
