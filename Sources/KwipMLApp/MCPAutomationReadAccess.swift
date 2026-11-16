import Foundation
import KwipMLCore

/// Finite client/project-owned snapshots of already persisted automation metadata.
@MainActor final class MCPAutomationReadAccess {
    private struct State {
        let client:String,project:String,root:URL,family:AutomationReadSnapshot.Family
        let snapshot:AutomationReadSnapshot.Snapshot,items:[AutomationReadSnapshot.Item],expires:Date
        let restricted:Bool,components:Set<String>
        var cursors:[String:Int]=[:]
    }
    private var states:[String:State]=[:]
    private var generation=0,opening=0
    var now:()->Date=Date.init
    func invalidate(){generation+=1;states.removeAll()}
    func read(client:String,project:String,root:URL,args:[String:JSONValue],authorize:@escaping (Set<String>)throws->Void)async throws->[String:JSONValue] {
        try authorize([])
        func refuse(_ code:String)throws->Never{throw MCPSettings.Refusal(code:code)}
        guard Set(args.keys).isSubset(of:["family","snapshotID","cursorID","recordID","limit"]),let name=args["family"]?.stringValue,let family=AutomationReadSnapshot.Family(rawValue:name) else{try refuse("automation_invalid_arguments")}
        for key in ["snapshotID","cursorID","recordID"] {
            if let value=args[key] {guard case .string(let id)=value,UUID(uuidString:id)?.uuidString==id else{try refuse("automation_invalid_arguments")}}
        }
        let limit:Int
        if let value=args["limit"] {guard case .integer(let n)=value,(1...100).contains(n) else{try refuse("automation_invalid_arguments")};limit=Int(n)}else{limit=50}
        guard args["recordID"]==nil || (args["snapshotID"] != nil && args["cursorID"]==nil && args["limit"]==nil) else{try refuse("automation_invalid_arguments")}
        let root=root.standardizedFileURL.resolvingSymlinksInPath(),epoch=generation
        states=states.filter{$0.value.expires>now()}
        let id:String
        if let supplied=args["snapshotID"]?.stringValue {
            guard let state=states[supplied],state.client==client,state.project==project,state.root==root,state.family==family else{try refuse("automation_snapshot_unavailable")}
            id=supplied
        }else{
            guard args["cursorID"]==nil,args["recordID"]==nil,states.count+opening<8 else{try refuse("automation_snapshot_required")}
            opening+=1;defer{opening-=1}
            let snapshot:AutomationReadSnapshot.Snapshot
            do {snapshot=try await AutomationReadSnapshot.load(root:root,projectID:project,family:family)}
            catch is CancellationError {throw CancellationError()}
            catch {try refuse("automation_source_unavailable")}
            try authorize([])
            guard epoch==generation else{try refuse("automation_snapshot_unavailable")}
            // Project filtering happened in the family projection. Component filtering
            // also precedes slicing, so pages never reveal denied rows/counts.
            var decisions:[Set<String>:Bool]=[:]
            let allowed=snapshot.items.filter {item in
                if let previous=decisions[item.components]{return previous}
                let accepted=(try? authorize(item.components)) != nil
                decisions[item.components]=accepted;return accepted
            }
            let components=allowed.reduce(into:Set<String>()){$0.formUnion($1.components)}
            id=UUID().uuidString
            states[id] = .init(client:client,project:project,root:root,family:family,snapshot:snapshot,items:allowed,expires:now().addingTimeInterval(300),restricted:allowed.count != snapshot.items.count || snapshot.omittedUnverifiableRecords,components:components)
        }
        guard var state=states[id] else{try refuse("automation_snapshot_unavailable")}
        // A grant change cannot silently reshape an existing snapshot's pagination.
        try authorize(state.components)
        let selected:[AutomationReadSnapshot.Item]
        var next:JSONValue = .null
        if let record=args["recordID"]?.stringValue {
            guard let item=state.items.first(where:{$0.record.id==record}) else{try refuse("automation_record_unavailable")}
            selected=[item]
        }else{
            let offset:Int
            if let cursor=args["cursorID"]?.stringValue {guard let found=state.cursors[cursor] else{try refuse("automation_cursor_unavailable")};offset=found}else{offset=0}
            selected=Array(state.items.dropFirst(offset).prefix(limit))
            if offset+selected.count<state.items.count {
                guard state.cursors.count<128 else{try refuse("automation_cursor_limit")}
                let cursor=UUID().uuidString;state.cursors[cursor]=offset+selected.count;next = .string(cursor)
            }
        }
        states[id]=state
        return ["snapshotID":.string(id),"family":.string(family.rawValue),"records":.array(try selected.map{try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode($0.record))}),"nextCursorID":next,"expiresInSeconds":.integer(Int64(max(0,state.expires.timeIntervalSince(now())))),"observedAt":.number(state.snapshot.observedAt.timeIntervalSinceReferenceDate),"sourceStatus":.string(state.snapshot.sourceStatus),"scope":.string(state.restricted ? "Authorized subset only; unavailable saved pins or component restrictions omit records.":"Project-scoped recorded metadata only."),"serviceReadiness":.string("notObserved"),"verification":.string("validatedLedgerAndSavedSourcePins"),"limitations":.string("Persisted status is not live service readiness or verified execution success. No payloads, credentials, History values, source documents or automation controls are exposed. Families are independently observed, not a cross-ledger atomic snapshot.")]
    }
}
