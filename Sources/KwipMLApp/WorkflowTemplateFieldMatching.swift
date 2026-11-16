import SwiftUI
import KwipMLCore

/// A reviewed header mapping becomes ordinary named Columns configuration, never a reader-position rename.
struct WorkflowTemplateFieldMatchPlan {
    let source:WorkflowNode
    let required:[String]
    let observed:[String]
    let previous:[String:String]
    let mapper:WorkflowNode?
    let mapperAddress:WorkflowNodeAddress
    private let sourceConfig:[String:JSONValue]
    private let expectedTypes:[String:JSONValue]

    init(document:WorkflowDocument,address:WorkflowNodeAddress,observed:[String]) throws {
        guard let origin=document.templateOrigin,
              !observed.isEmpty,observed.count<=128,Set(observed).count==observed.count,
              observed.allSatisfy({!$0.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty && $0.count<=200}),
              let source=document.nodes.first(where:{$0.address==address}),
              case .dataOperator("kwip.data.sourceCSV",let config)=source.payload,
              (config["format"]?.stringValue ?? "csv")=="csv",
              (config["headerMode"]?.stringValue ?? "firstRecord")=="firstRecord" else {
            throw DataFlowContractError("Choose and inspect a CSV with named headers before matching fields.")
        }
        for key in ["schema","columns","newColumns"] {
            switch config[key] {case nil,.null,.array([]),.object([:]):break
            default:throw DataFlowContractError("This source has advanced reader projection or naming settings. Review them in Configuration first.")}
        }
        let original=try WorkflowTemplates.instantiate(id:origin.id)
        guard original.templateOrigin==origin,
              let base=original.nodes.first(where:{$0.address==address}),
              case .dataOperator("kwip.data.sourceCSV",let baseConfig)=base.payload else {
            throw DataFlowContractError("This input is not part of the original template. Configure it on the canvas.")
        }
        var fields:[String:JSONValue]=[:]
        if case .object(let types)=baseConfig["schemaOverrides"] {fields=types}
        let baseTypes=fields
        if fields.isEmpty {
            for edge in original.edges where edge.from.address==address {
                guard let next=original.nodes.first(where:{$0.address==edge.to.address}),
                      case .dataOperator("kwip.data.columns",let c)=next.payload,
                      case .array(let rows)=c["rows"] else {continue}
                for row in rows {if case .object(let r)=row,let name=r["source"]?.stringValue {fields[name] = .string("String")}}
            }
        }
        guard !fields.isEmpty,fields.count<=128 else {throw DataFlowContractError("No explicit template field contract is available for this input.")}
        let names=fields.keys.sorted(),mapAddress=WorkflowNodeAddress(domain:.data,localID:"template-fields-"+address.localID)
        let existing=document.nodes.first(where:{$0.address==mapAddress})
        var previous=Dictionary(uniqueKeysWithValues:names.map{($0,$0)})
        if let existing {
            guard case .dataOperator("kwip.data.columns",let c)=existing.payload,
                  Set(c.keys)==["rows","newColumns","trimText","castFailure","missingPolicy"],
                  c["newColumns"] == .string("keep"),c["trimText"] == .bool(false),
                  c["castFailure"] == .string("error"),c["missingPolicy"] == .string("preserve"),
                  case .array(let rows)=c["rows"],rows.count==names.count,
                  Set(document.edges.filter{$0.from.address==address}.map(\.to)) == [.init(address:mapAddress,port:"table")],
                  document.edges.filter({$0.to.address==mapAddress}).count==1 else {
                throw DataFlowContractError("The field-matching block has been edited. Review its Columns configuration on the canvas.")
            }
            previous=[:]
            for row in rows {
                guard case .object(let r)=row,let input=r["source"]?.stringValue,let output=r["outputName"]?.stringValue,
                      names.contains(output),previous[output]==nil,r["included"] == .bool(true),r["targetType"] == .string("keep") else {
                    throw DataFlowContractError("The field-matching block has been edited. Review it on the canvas.")
                }
                previous[output]=input
            }
            guard Set(previous.values).count==names.count else {throw DataFlowContractError("Field matches must use distinct input columns.")}
        }
        self.source=source;required=names;self.observed=observed;self.previous=previous
        mapper=existing;mapperAddress=mapAddress;sourceConfig=config;expectedTypes=baseTypes
    }

    var initial:[String:String] {
        Dictionary(uniqueKeysWithValues:required.map {name in
            (name,observed.contains(name) ? name : (observed.contains(previous[name] ?? "") ? previous[name]! : ""))
        })
    }
    func commands(in document:WorkflowDocument,matches:[String:String]) throws -> [WorkflowCommand] {
        guard document.nodes.first(where:{$0.address==source.address})==source,
              Set(matches.keys)==Set(required),matches.values.allSatisfy({observed.contains($0)}),
              Set(matches.values).count==required.count,
              required.allSatisfy({!observed.contains($0) || matches[$0]==$0}) else {
            throw DataFlowContractError("Choose one distinct input column for every required field. Exact header matches stay unchanged; missing business fields cannot be invented.")
        }
        let outputNames=observed.map {input in matches.first(where:{$0.value==input})?.key ?? input}
        guard Set(outputNames).count==outputNames.count else {throw DataFlowContractError("These matches would create duplicate column names.")}
        if (mapper==nil && matches.allSatisfy({$0.key==$0.value})) || (mapper != nil && matches==previous) {return []}
        guard document.nodes.first(where:{$0.address==mapperAddress})==mapper else {throw DataFlowContractError("The field-matching block changed. Reopen matching.")}
        var commands:[WorkflowCommand]=[],nextSource=sourceConfig
        if case .object(let existing)=sourceConfig["schemaOverrides"],!existing.isEmpty {
            var remaining=existing,newTypes:[String:JSONValue]=[:]
            for name in required {if let type=remaining.removeValue(forKey:previous[name] ?? name) ?? expectedTypes[name] {newTypes[matches[name]!] = type}}
            guard Set(remaining.keys).isDisjoint(with:Set(newTypes.keys)) else {throw DataFlowContractError("A custom type override conflicts with the chosen field. Review source Configuration.")}
            remaining.merge(newTypes){_,new in new};nextSource["schemaOverrides"] = .object(remaining)
        }
        if nextSource != sourceConfig {commands.append(.configure(address:source.address,config:nextSource))}
        let rows=observed.compactMap {input -> JSONValue? in
            guard let output=matches.first(where:{$0.value==input})?.key else {return nil}
            return .object(["source":.string(input),"outputName":.string(output),"included":.bool(true),"targetType":.string("keep")])
        }
        let mapping:[String:JSONValue] = ["rows":.array(rows),"castFailure":.string("error"),"missingPolicy":.string("preserve"),"newColumns":.string("keep"),"trimText":.bool(false)]
        if let mapper {
            guard case .dataOperator(_,let old)=mapper.payload else {throw DataFlowContractError("Review the field-matching block.")}
            if old != mapping {commands.append(.configure(address:mapperAddress,config:mapping))}
        } else {
            let outgoing=document.edges.filter{$0.from.address==source.address}
            guard !outgoing.isEmpty,outgoing.allSatisfy({$0.kind == .table && $0.from.port=="table"}) else {throw DataFlowContractError("Connect this input to its template workflow before matching fields.")}
            let right=outgoing.compactMap {edge in document.nodes.first(where:{$0.address==edge.to.address})?.x}.filter{$0>source.x}.min()
            let x=right.map{(source.x+$0)/2} ?? source.x+300
            commands.append(.insert(.init(address:mapperAddress,name:"Match input fields",x:x,y:source.y,payload:.dataOperator(operatorID:"kwip.data.columns",config:mapping))))
            commands.append(.connect(from:.init(address:source.address,port:"table"),to:.init(address:mapperAddress,port:"table")))
            commands += outgoing.map{.reconnect(edgeID:$0.id,from:.init(address:mapperAddress,port:"table"),to:$0.to)}
        }
        return commands
    }
}

@MainActor final class WorkflowTemplateFieldMatchSession:ObservableObject {
    let owner:WorkflowAuthoringStore
    let sources:any WorkflowDataSources
    let plan:WorkflowTemplateFieldMatchPlan
    let lease:WorkflowEditorLease
    let contextID=UUID()
    private let sourceSHA:String
    @Published private(set) var choices:[String:String]
    @Published private(set) var message:String?
    var dirty:Bool {choices != plan.previous}
    var validationMessage:String? {guard current() else{return "The source or selection changed. Revert, inspect the source and reopen matching."};do {_ = try plan.commands(in:owner.document,matches:choices);return nil}catch{return error.localizedDescription}}
    var canApply:Bool {current() && dirty && (try? plan.commands(in:owner.document,matches:choices)) != nil}
    init(step:WorkflowTemplateSetupStep,owner:WorkflowAuthoringStore,sources:any WorkflowDataSources) throws {
        guard WorkflowTemplateSetupPlan.focus(step,owner:owner,sources:sources,lease:owner.lease()),
              let preview=sources.sourceInspectionPreview(for:step.address) else {throw DataFlowContractError("Inspect this source first, then match its actual column headers.")}
        let plan=try WorkflowTemplateFieldMatchPlan(document:owner.document,address:step.address,observed:preview.columns)
        self.plan=plan;self.owner=owner;self.sources=sources;lease=owner.lease();sourceSHA=preview.sourceSHA256;choices=plan.initial
        guard report() else {throw DataFlowContractError("Use the current idle source before matching fields.")}
    }
    private func current()->Bool {
        owner.isCurrent(lease) && owner.selection.nodeAddresses == [plan.source.address] && owner.selection.edgeIDs.isEmpty && !owner.isBusy && !sources.isBusy && sources.projectID==owner.projectID && sources.sessionID==owner.sessionID &&
        sources.sourceInspectionPreview(for:plan.source.address).map{$0.sourceSHA256==sourceSHA && $0.columns==plan.observed} == true
    }
    @discardableResult private func report()->Bool {
        owner.setInspectorEditPending(contextID:contextID,address:plan.source.address,key:"templateFieldMatches",pending:dirty,lease:lease)
    }
    func choose(_ field:String,_ column:String) {
        guard current(),plan.required.contains(field),plan.observed.contains(column) || column.isEmpty,
              !plan.observed.contains(field) || column==field else {message="The source or selection changed. Revert and inspect it again.";return}
        var next=choices;next[field]=column
        guard owner.setInspectorEditPending(contextID:contextID,address:plan.source.address,key:"templateFieldMatches",pending:next != plan.previous,lease:lease) else {message="Return to this source or revert the field draft.";return}
        choices=next;message=nil
    }
    @discardableResult func apply()->Bool {
        guard current() else {message="The source changed. Revert, inspect the new file and reopen matching.";return false}
        do {
            let commands=try plan.commands(in:owner.document,matches:choices)
            owner.clearDataEdits(contextID:contextID)
            guard !owner.hasPendingInspectorEdits else { _ = report();throw DataFlowContractError("Apply or revert the other pending configuration first.")}
            if !commands.isEmpty,owner.applyBatch(commands,lease:lease)==nil {_ = report();throw DataFlowContractError(owner.error ?? "No field matches were applied.")}
            message=nil;return true
        } catch {message=error.localizedDescription;return false}
    }
    func discard(){owner.clearDataEdits(contextID:contextID)}
}

struct WorkflowTemplateFieldMatchView:View {
    @ObservedObject var session:WorkflowTemplateFieldMatchSession
    let done:(Bool)->Void
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            Text("Match your column names").font(.system(size:12,weight:.semibold))
            Text("Choose which column means each missing field. Values stay unchanged; a Columns block records your choices on the canvas.").font(.caption).foregroundStyle(MLTheme.muted)
            ForEach(session.plan.required.filter{!session.plan.observed.contains($0)},id:\.self) {field in
                VStack(alignment:.leading,spacing:4) {
                    Text(field).font(.caption.weight(.medium))
                    FlatChoiceControl(title:"Input column for "+field,selection:session.choices[field] ?? "",options:session.plan.observed.map{.init(id:$0,title:$0)},placeholder:"Choose a column") {session.choose(field,$0)}
                }
            }
            let exact=session.plan.required.filter{session.plan.observed.contains($0)}
            if !exact.isEmpty {Text("\(exact.count) fields already match their exact headers.").font(.caption).foregroundStyle(MLTheme.muted)}
            if let message=session.message ?? session.validationMessage {Text(message).font(.caption).foregroundStyle(MLTheme.rose)}
            HStack {
                Button("Apply field matches") {if session.apply(){done(true)}}.disabled(!session.canApply)
                    .accessibilityIdentifier("template.fields.apply")
                Button(session.dirty ? "Revert":"Done") {session.discard();done(false)}.accessibilityIdentifier("template.fields.revert")
            }.buttonStyle(FlatButtonStyle())
        }.padding(12).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
            .onDisappear {session.discard()}
    }
}
