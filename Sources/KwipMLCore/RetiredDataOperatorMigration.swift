import Foundation

/// Pure one-time authoring upgrade. Never reads sources or changes historical evidence.
/// Unsupported semantics refuse the whole transaction before the caller saves anything.
public enum RetiredDataOperatorMigration {
    public static let retired:Set<String>=Set("missing standardScalerFit imputerFit oneHotEncoderFit transform calculate regexExtract regexReplace split groupSplit temporalSplit selectRename cast filter filterExpression regexMatch".split(separator:" ").map {"kwip.data."+$0})
    private static let fits:Set<String>=["kwip.data.standardScalerFit","kwip.data.imputerFit","kwip.data.oneHotEncoderFit"]
    public static func requiresUpgrade(_ document:WorkflowDocument)->Bool {document.nodes.contains {if case .dataOperator(let op,_)=$0.payload {return retired.contains(op)};return false}}
    public static func requireCurrent(_ document:WorkflowDocument)throws {
        guard !requiresUpgrade(document) else {
            throw DataFlowContractError("This project contains retired data tools. Upgrade exact supported settings or repair the retained components with current Columns, Filter, Formula, Partition or Preprocess before running or exporting code. Original source is unchanged.")
        }
    }
    public typealias Schema=[[String:JSONValue]]
    private struct Upgrade {var graph:DataFlowDocument;var rewrites:[DataFlowEndpoint:DataFlowEndpoint];var removed:Set<String>;var fused:Set<String>}
    private static func refuse(_ id:String,_ message:String)throws->Never {
        throw DataFlowContractError("Retired data tool \(id): \(message) Original source is unchanged; repair with current Columns, Filter, Formula, Partition or Preprocess before saving or running.")
    }
    private static func string(_ value:JSONValue?,_ id:String)throws->String {
        guard case .string(let v)=value else {try refuse(id,"A required text control is invalid.")};return v
    }
    private static func strings(_ value:JSONValue?,_ id:String)throws->[String] {
        guard case .array(let values)=value,!values.isEmpty,values.count<=128 else {try refuse(id,"Column selections must be bounded and nonempty.")}
        let names=try values.map {try string($0,id)}
        guard Set(names).count==names.count,names.allSatisfy({!$0.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty && $0.count<=200}) else {try refuse(id,"Column names are invalid or repeated.")};return names
    }
    private static func keys(_ c:[String:JSONValue],_ required:Set<String>,_ optional:Set<String>=[],_ id:String)throws {
        guard required.isSubset(of:Set(c.keys)),Set(c.keys).isSubset(of:required.union(optional)) else {try refuse(id,"Legacy controls are incomplete or unknown.")}
    }
    private static func quoted(_ text:String)throws->String {
        let encoder=JSONEncoder();encoder.outputFormatting=[.withoutEscapingSlashes]
        return String(decoding:try encoder.encode(text),as:UTF8.self)
    }
    private static func write(_ c:[String:JSONValue],_ name:String,_ schema:Schema?,_ id:String)throws->JSONValue {
        guard case .bool(let replace)=c["replaceExisting"] else {try refuse(id,"Replace existing must be explicit.")}
        if !replace {return .string("new")}
        guard let schema else {try refuse(id,"Replace existing requires a known complete input schema.")}
        return .string(schema.contains {$0["name"] == .string(name)} ? "replace":"new")
    }
    private static func configuration(_ op:String,_ c:[String:JSONValue],_ id:String,_ schema:Schema?)throws->(String,[String:JSONValue]) {
        switch op {
        case "kwip.data.split":try refuse(id,"NumPy two-way shuffled membership differs from current three-way Partition. Choose and review a new split explicitly.")
        case "kwip.data.filter":try refuse(id,"The legacy single-output filter stops on an empty result; review the intended current Filter branches and empty-result behavior explicitly.")
        case "kwip.data.selectRename","kwip.data.cast":
            var rows:[JSONValue]=[]
            if op == "kwip.data.selectRename" {
                try keys(c,["columns","rename"],[],id)
                let selected=try strings(c["columns"],id)
                guard case .object(let names)=c["rename"],Set(names.keys).isSubset(of:Set(selected)) else {try refuse(id,"Rename controls do not match selected sources.")}
                rows=try selected.map {name in .object(["source":.string(name),"included":true,"outputName":.string(try names[name].map {try string($0,id)} ?? name),"targetType":.string("keep")])}
            } else {
                try keys(c,["columns"],["missingPolicy"],id)
                guard let schema,!schema.isEmpty,case .object(let casts)=c["columns"],!casts.isEmpty else {try refuse(id,"Cast requires the known complete input schema to preserve original column order.")}
                let names=try schema.map {try string($0["name"],id)}
                guard Set(names).count==names.count,Set(casts.keys).isSubset(of:Set(names)) else {try refuse(id,"Cast source is absent or the supplied schema is ambiguous.")}
                for type in casts.values {guard ["string","int64","float64","bool"].contains(try string(type,id)) else {try refuse(id,"Unsupported legacy cast type.")}}
                rows=names.map {.object(["source":.string($0),"included":true,"outputName":.string($0),"targetType":casts[$0] ?? .string("keep")])}
            }
            return ("kwip.data.columns",["rows":.array(rows),"castFailure":.string("error"),"trimText":false,"newColumns":.string("drop"),"missingPolicy":c["missingPolicy"] ?? .string("error")])
        case "kwip.data.missing":
            try keys(c,["columns","policy"],["value"],id)
            let policy=try string(c["policy"],id)
            guard ["error","drop","fill"].contains(policy),(c["value"] != nil)==(policy=="fill") else {try refuse(id,"Invalid missing-value controls.")}
            if policy=="drop" {try refuse(id,"Legacy Drop stops when every observation disappears; review current Preprocess empty-table behavior explicitly.")}
            let rules=try strings(c["columns"],id).map {name -> PreprocessConfiguration.Column in
                var rule=PreprocessConfiguration.Column(column:name);rule.missing=policy=="error" ? "error":"constant"
                if policy=="fill" {
                    guard let value=c["value"],let dtype=schema?.first(where:{$0["name"] == .string(name)})?["dtype"]?.stringValue,PreprocessConfiguration.matches(value,dtype:dtype) else {try refuse(id,"A fixed replacement needs a known matching dtype; implicit literal casts cannot be guessed.")}
                    rule.value=value
                };return rule
            }
            return ("kwip.data.preprocess",PreprocessConfiguration(mode:"table",columns:rules).values)
        case "kwip.data.groupSplit":
            try keys(c,["groupBy","nullKeys","seed","allocationUnit","validationPercent","testPercent"],[],id)
            guard c["nullKeys"] == .string("error"),c["allocationUnit"] == .string("groups") else {try refuse(id,"Unsupported group allocation semantics.")}
            var values=c;values.removeValue(forKey:"nullKeys");values.removeValue(forKey:"allocationUnit");values["strategy"] = .string("entity")
            return ("kwip.data.partition",values)
        case "kwip.data.temporalSplit":
            try keys(c,["timeColumn","timeEncoding","trainingEnd","validationEnd","embargoMilliseconds","entityColumns","entityPolicy"],[],id)
            var values=c;values["strategy"] = .string("time");return ("kwip.data.partition",values)
        case "kwip.data.filterExpression","kwip.data.regexMatch":
            var values:[String:JSONValue]=["mode":.string("expression"),"conditions":.array([]),"match":.string("all"),"expression":.string(""),"column":.string("value"),"pattern":.string(""),"matchMode":.string("search"),"flags":.array([]),"nullPolicy":.string("error")]
            if op=="kwip.data.filterExpression" {try keys(c,["expression","nullPolicy"],[],id);values.merge(c){_,v in v}}
            else {try keys(c,["column","pattern","flags","nullPolicy","matchMode"],[],id);values["mode"] = .string("pattern");values.merge(c){_,v in v}}
            return ("kwip.data.filterRows",values)
        case "kwip.data.calculate","kwip.data.regexExtract","kwip.data.regexReplace":
            let expression:String;let outputs:[JSONValue]
            if op=="kwip.data.calculate" {
                try keys(c,["expression","outputColumn","dtype","replaceExisting","nullPolicy"],[],id)
                let name=try string(c["outputColumn"],id);expression=try string(c["expression"],id)
                outputs=[.object(["name":.string(name),"dtype":c["dtype"]!,"write":try write(c,name,schema,id)])]
            } else {
                let source=try string(c["column"],id)
                guard !source.contains("]"),!source.contains("\n") else {try refuse(id,"Source column cannot be represented exactly by Formula syntax.")}
                guard case .array(let flags)=c["flags"] else {try refuse(id,"Regex flags are invalid.")}
                let flagText=try flags.map {try string($0,id)}.joined(separator:"|")
                if op=="kwip.data.regexExtract" {
                    try keys(c,["column","pattern","flags","nullPolicy","groups","noMatchPolicy","replaceExisting"],[],id)
                    guard case .array(let groups)=c["groups"],!groups.isEmpty,groups.count<=32 else {try refuse(id,"Regex groups are invalid.")}
                    var options=[try string(c["pattern"],id),"search",try string(c["noMatchPolicy"],id),flagText];var result:[JSONValue]=[]
                    for group in groups {
                        guard case .object(let g)=group,Set(g.keys)==["group","output"] else {try refuse(id,"Regex group controls are invalid.")}
                        let name=try string(g["output"],id);options.append(try string(g["group"],id));result.append(.object(["name":.string(name),"dtype":.string("string"),"write":try write(c,name,schema,id)]))
                    }
                    expression="regex_extract(["+source+"], "+(try options.map(quoted)).joined(separator:", ")+")";outputs=result
                } else {
                    try keys(c,["column","pattern","flags","nullPolicy","replacement","replaceMode","outputColumn","replaceExisting"],[],id)
                    let name=try string(c["outputColumn"],id)
                    let options=[try string(c["pattern"],id),try string(c["replacement"],id),try string(c["replaceMode"],id),flagText]
                    expression="regex_replace(["+source+"], "+(try options.map(quoted)).joined(separator:", ")+")"
                    outputs=[.object(["name":.string(name),"dtype":.string("string"),"write":try write(c,name,schema,id)])]
                }
            }
            return ("kwip.data.formula",["operations":.array([.object(["id":.string("upgraded-operation"),"expression":.string(expression),"outputs":.array(outputs),"nullPolicy":c["nullPolicy"]!])])])
        default:try refuse(id,"Fitted transforms require one exactly fusible partition family.")
        }
    }
    private static func upgrade(_ original:DataFlowDocument,_ schemas:[String:Schema])throws->Upgrade {
        try DataFlowCodec.validate(original)
        var graph=original;var removed=Set<String>(),fused=Set<String>(),rewrites:[DataFlowEndpoint:DataFlowEndpoint]=[:]
        // Traverse dependencies without changing the saved node-array order.
        // A downstream fit becomes provable only after its upstream transforms
        // have been fused and their role endpoints rewritten.
        var remaining=Set(original.nodes.map(\.id)),completed=Set<String>(),orderedFits:[DataFlowNode]=[]
        while !remaining.isEmpty {
            let ready=original.nodes.filter {node in remaining.contains(node.id) && original.edges.filter {$0.to.node==node.id}.allSatisfy {completed.contains($0.from.node)}}
            guard !ready.isEmpty else {try refuse("graph","A dependency cycle prevents exact fitted-chain fusion.")}
            orderedFits += ready.filter {fits.contains($0.operatorID)}
            completed.formUnion(ready.map(\.id));remaining.subtract(ready.map(\.id))
        }
        for old in orderedFits {
            let incoming=graph.edges.filter {$0.to == .init(node:old.id,port:"training")}
            let consumers=graph.edges.filter {$0.from == .init(node:old.id,port:"state")}
            guard incoming.count==1,!consumers.isEmpty else {try refuse(old.id,"Fit/apply fusion requires one Training input and paired transform consumers.")}
            let source=incoming[0].from
            guard source.port=="training",let family=graph.nodes.first(where:{$0.id==source.node}),["kwip.data.partition","kwip.data.groupSplit","kwip.data.temporalSplit","kwip.data.preprocess"].contains(family.operatorID) else {try refuse(old.id,"The direct Training partition family cannot be proved.")}
            var roles:[String:DataFlowEndpoint]=[:]
            for edge in consumers {
                guard edge.to.port=="state",let target=graph.nodes.first(where:{$0.id==edge.to.node}),target.operatorID=="kwip.data.transform",target.config.isEmpty else {try refuse(old.id,"The fitted state has an unpaired consumer.")}
                let inputs=graph.edges.filter {$0.to == .init(node:target.id,port:"table")}
                guard inputs.count==1,inputs[0].from.node==source.node,["training","validation","test"].contains(inputs[0].from.port) else {try refuse(old.id,"Paired transforms must consume direct roles of the same partition family.")}
                let role=inputs[0].from.port;roles[role]=inputs[0].from;removed.insert(target.id);rewrites[.init(node:target.id,port:"table")] = .init(node:old.id,port:role)
            }
            guard roles["training"] != nil else {try refuse(old.id,"An explicit Training transform is required.")}
            let required:Set<String>=old.operatorID=="kwip.data.standardScalerFit" ? ["columns"]:old.operatorID=="kwip.data.imputerFit" ? ["columns","strategy","emptyPolicy"]:["columns","vocabularyOrder","unknownPolicy","maxCategories"]
            try keys(old.config,required,old.operatorID=="kwip.data.imputerFit" ? ["emptyValue","tiePolicy"]:[],old.id)
            let rules=try strings(old.config["columns"],old.id).map {name -> PreprocessConfiguration.Column in
                var rule=PreprocessConfiguration.Column(column:name)
                if old.operatorID=="kwip.data.standardScalerFit" {rule.scaling="standard"}
                else if old.operatorID=="kwip.data.imputerFit" {rule.missing=try string(old.config["strategy"],old.id);rule.emptyPolicy=try string(old.config["emptyPolicy"],old.id);rule.emptyValue=old.config["emptyValue"] ?? 0;rule.tiePolicy=try old.config["tiePolicy"].map {try string($0,old.id)} ?? "error"}
                else {rule.encoding="oneHot";rule.vocabularyOrder=try string(old.config["vocabularyOrder"],old.id);rule.unknownPolicy=try string(old.config["unknownPolicy"],old.id);guard let maximum=old.config["maxCategories"]?.integerValue,let value=Int(exactly:maximum) else {try refuse(old.id,"Category maximum is invalid.")};rule.maxCategories=value};return rule
            }
            let config=PreprocessConfiguration(mode:"partitions",columns:rules);try config.validate()
            let index=graph.nodes.firstIndex {$0.id==old.id}!;graph.nodes[index].operatorID="kwip.data.preprocess";graph.nodes[index].config=config.values;fused.insert(old.id)
            graph.edges.removeAll {$0.to.node==old.id || removed.contains($0.to.node)}
            graph.edges += roles.keys.sorted().map {.init(from:roles[$0]!,to:.init(node:old.id,port:$0))}
            for i in graph.edges.indices {if let endpoint=rewrites[graph.edges[i].from] {graph.edges[i].from=endpoint}}
        }
        for index in graph.nodes.indices {
            let node=graph.nodes[index]
            guard retired.contains(node.operatorID),!removed.contains(node.id) else {continue}
            let (op,config)=try configuration(node.operatorID,node.config,node.id,schemas[node.id]);graph.nodes[index].operatorID=op;graph.nodes[index].config=config
        }
        graph.nodes.removeAll {removed.contains($0.id)}
        for i in graph.edges.indices {if let endpoint=rewrites[graph.edges[i].from] {graph.edges[i].from=endpoint}}
        try DataFlowCodec.validate(graph)
        return .init(graph:graph,rewrites:rewrites,removed:removed,fused:fused)
    }
    public static func migrate(_ graph:DataFlowDocument,knownInputSchemas:[String:Schema]=[:])throws->DataFlowDocument {try upgrade(graph,knownInputSchemas).graph}
    public static func migrate(_ document:WorkflowDocument,knownInputSchemas:[WorkflowNodeAddress:Schema]=[:])throws->WorkflowDocument {
        let dataNodes=document.nodes.compactMap {node->DataFlowNode? in guard case .dataOperator(let op,let config)=node.payload else {return nil};return .init(id:node.address.localID,operatorID:op,config:config)}
        guard dataNodes.contains(where:{retired.contains($0.operatorID)}) else {return document}
        let edges=document.edges.filter {$0.from.address.domain == .data && $0.to.address.domain == .data}.map {DataFlowEdge(from:.init(node:$0.from.address.localID,port:$0.from.port),to:.init(node:$0.to.address.localID,port:$0.to.port))}
        let schemas=Dictionary(uniqueKeysWithValues:knownInputSchemas.filter {$0.key.domain == .data}.map {($0.key.localID,$0.value)})
        let result=try upgrade(.init(nodes:dataNodes,edges:edges),schemas);var output=document
        // Only these exact upgrades change the declared role-output kind.
        // Preserve all unrelated kinds, including already-invalid declarations.
        let tablePartitions=Set(dataNodes.filter {node in
            ["kwip.data.groupSplit","kwip.data.temporalSplit"].contains(node.operatorID) && result.graph.nodes.contains {$0.id==node.id && $0.operatorID=="kwip.data.partition"}
        }.map(\.id))
        func upgradedKind(_ endpoint:WorkflowEndpoint,_ original:WorkflowEdgeKind)->WorkflowEdgeKind {
            endpoint.address.domain == .data && tablePartitions.contains(endpoint.address.localID) && ["training","validation","test"].contains(endpoint.port) && original == .dataset ? .table:original
        }
        output.nodes.removeAll {$0.address.domain == .data && result.removed.contains($0.address.localID)}
        for i in output.nodes.indices where output.nodes[i].address.domain == .data {
            if let node=result.graph.nodes.first(where:{$0.id==output.nodes[i].address.localID}) {output.nodes[i].payload = .dataOperator(operatorID:node.operatorID,config:node.config)}
        }
        output.edges=document.edges.filter {!($0.from.address.domain == .data && $0.to.address.domain == .data) && !($0.to.address.domain == .data && result.removed.contains($0.to.address.localID))}
        for i in output.edges.indices where output.edges[i].from.address.domain == .data {
            output.edges[i].kind=upgradedKind(output.edges[i].from,output.edges[i].kind)
            if let endpoint=result.rewrites[.init(node:output.edges[i].from.address.localID,port:output.edges[i].from.port)] {output.edges[i].from = .init(address:.init(domain:.data,localID:endpoint.node),port:endpoint.port)}
        }
        var originalEdges:[DataFlowEdge:WorkflowEdge]=[:]
        for edge in document.edges where edge.from.address.domain == .data && edge.to.address.domain == .data {
            var from=DataFlowEndpoint(node:edge.from.address.localID,port:edge.from.port)
            if let rewritten=result.rewrites[from] {from=rewritten}
            originalEdges[.init(from:from,to:.init(node:edge.to.address.localID,port:edge.to.port))]=edge
        }
        var used=Set(document.edges.map(\.id));var serial=0
        for edge in result.graph.edges {
            let original=originalEdges[edge];let id:String
            if let original {id=original.id}
            else {while used.contains("data-upgrade-\(serial)") {serial+=1};id="data-upgrade-\(serial)";used.insert(id);serial+=1}
            let from=WorkflowEndpoint(address:.init(domain:.data,localID:edge.from.node),port:edge.from.port)
            output.edges.append(.init(id:id,from:from,to:.init(address:.init(domain:.data,localID:edge.to.node),port:edge.to.port),kind:upgradedKind(from,original?.kind ?? .table)))
        }
        // Every retained node keeps its original address/name/geometry.
        return output
    }
}
