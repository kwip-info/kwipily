import Foundation
import KwipMLCore

/// Client-owned metadata snapshots; retained artifact verification stays in the family reader.
@MainActor final class MCPHistoryReadAccess {
    private struct State {
        let client:String,project:String,root:URL,family:WorkflowRetainedHistoryReader.Family
        let snapshot:WorkflowRetainedHistoryReader.Snapshot,expires:Date
        var cursors:[String:WorkflowRetainedHistoryReader.Cursor]=[:]
        var exposed:[String:WorkflowRetainedHistoryReader.Metadata]=[:]
    }
    private var states:[String:State]=[:]
    private var generation=0
    private var opening=0
    var now:()->Date=Date.init
    func invalidate(){generation+=1;states.removeAll()}
    private func refusal(_ code:String)->MCPSettings.Refusal {.init(code:code)}
    func read(client:String,project:String,root:URL,args:[String:JSONValue],includeValues:Bool,
              authorize:@escaping (Set<String>,Bool)throws->Void)async throws->[String:JSONValue] {
        try authorize([],false)
        guard Set(args.keys).isSubset(of:["operation","family","snapshotID","cursorID","recordID","limit"]),
              let operation=args["operation"]?.stringValue,["list","inspect"].contains(operation),
              let rawFamily=args["family"]?.stringValue else {throw refusal("history_invalid_arguments")}
        guard let family=WorkflowRetainedHistoryReader.Family(rawValue:rawFamily) else {throw refusal("history_family_unsupported")}
        for key in ["snapshotID","cursorID","recordID"] {
            if let value=args[key] {guard case .string(let token)=value,UUID(uuidString:token)?.uuidString==token else {throw refusal("history_invalid_arguments")}}
        }
        if let value=args["limit"] {guard case .integer(let n)=value,(1...100).contains(n) else {throw refusal("history_invalid_arguments")}}

        let root=root.standardizedFileURL.resolvingSymlinksInPath(),epoch=generation
        states=states.filter{$0.value.expires>now()}
        let id:String
        if let supplied=args["snapshotID"]?.stringValue {
            guard let state=states[supplied],state.client==client,state.project==project,state.root==root,state.family==family else {throw refusal("history_snapshot_unavailable")}
            id=supplied
        } else {
            guard operation=="list",args["cursorID"]==nil,args["recordID"]==nil,states.count+opening<8 else {throw refusal("history_snapshot_required")}
            opening+=1
            defer {opening-=1}
            let snapshot:WorkflowRetainedHistoryReader.Snapshot
            do {snapshot=try await WorkflowRetainedHistoryReader.open(root:root,projectID:project,family:family)}
            catch is CancellationError {throw CancellationError()}
            catch {throw refusal("history_snapshot_unavailable")}
            try authorize([],false)
            guard epoch==generation else {throw refusal("history_snapshot_unavailable")}
            id=UUID().uuidString
            states[id] = .init(client:client,project:project,root:root,family:family,snapshot:snapshot,expires:now().addingTimeInterval(300))
        }
        guard var state=states[id] else {throw refusal("history_snapshot_unavailable")}
        if operation=="inspect" {
            guard includeValues else {throw refusal("captured_values_required")}
            guard args["cursorID"]==nil,args["limit"]==nil,let record=args["recordID"]?.stringValue,let metadata=state.exposed[record],!metadata.components.isEmpty else {throw refusal("history_record_unavailable")}
            try authorize(metadata.components,true)
            let verified:WorkflowRetainedHistoryReader.Verified
            do {verified=try await WorkflowRetainedHistoryReader.inspect(root:root,projectID:project,metadata:metadata)}
            catch is CancellationError {throw CancellationError()}
            catch {throw refusal("history_evidence_unavailable")}
            try authorize(metadata.components,true)
            guard epoch==generation,(states[id]?.expires ?? .distantPast)>now() else {throw refusal("history_snapshot_unavailable")}
            return ["snapshotID":.string(id),"recordID":.string(record),"family":.string(family.rawValue),"verification":.string("verifiedRetainedArtifacts"),"provenance":try encoded(verified.provenance),"scope":.string("Verified captured operation only. No archived values returned by this endpoint; no current graph, original source or exported destination verification.")]
        }
        guard args["recordID"]==nil else {throw refusal("history_invalid_arguments")}
        let limit:Int
        if let value=args["limit"] {guard case .integer(let n)=value,(1...100).contains(n) else {throw refusal("history_invalid_arguments")};limit=Int(n)} else {limit=50}
        let cursor:WorkflowRetainedHistoryReader.Cursor?
        if let token=args["cursorID"]?.stringValue {guard let found=state.cursors[token] else {throw refusal("history_cursor_unavailable")};cursor=found}
        else {cursor=nil}
        let page=try state.snapshot.page(cursor:cursor,limit:limit)
        var records:[JSONValue]=[]
        for metadata in page.records where !metadata.components.isEmpty {
            guard (try? authorize(metadata.components,false)) != nil else {continue}
            state.exposed[metadata.operationID]=metadata
            records.append(try encoded(metadata))
        }
        var next:JSONValue = .null
        if let cursor=page.nextCursor {
            guard state.cursors.count<128 else {throw refusal("history_cursor_limit")}
            let token=UUID().uuidString;state.cursors[token]=cursor;next = .string(token)
        }
        states[id]=state
        return ["snapshotID":.string(id),"family":.string(family.rawValue),"records":.array(records),"nextCursorID":next,"expiresInSeconds":.integer(Int64(max(0,state.expires.timeIntervalSince(now())))),"verification":.string("recordedReceiptOnly"),"supportedFamilies":.array(WorkflowRetainedHistoryReader.Family.allCases.map{.string($0.rawValue)}),"scope":.string("Only authorized receipt metadata for this family. Unsupported, missing-identity, invalid or denied records are not returned. This is not a complete all-family History inventory.")]
    }
    private func encoded<T:Encodable>(_ value:T)throws->JSONValue {try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(value))}
}
