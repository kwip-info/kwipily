import Foundation
import KwipMLCore

struct BagPreparedObservation: Equatable {
    let input: BagObservationInput
    let targets: [String: JSONValue]
    let binaryLabel: Int?
}
struct VerifiedBagPartition: Equatable {
    let count: Int
    let rowIDs: [String]
    let tableSHA256: String
    let lineage: [String: JSONValue]
    let prefix: [BagPreparedObservation]
    let sourceIdentity: [String: JSONValue]
    var itemCounts: [Int] = []
}
struct VerifiedBagPreparation: Equatable {
    let selection: WorkflowPreparedSelection
    let training: VerifiedBagPartition
    let validation: VerifiedBagPartition
    let preparationIdentity: [String: JSONValue]
}
enum WorkflowBagPreparedEvidence {
    static func require(_ condition: Bool, _ text: String) throws { if !condition { throw WorkflowPhaseError(text) } }
    static func object(_ value: JSONValue?) throws -> [String: JSONValue] {
        guard case .object(let fields) = value else { throw WorkflowPhaseError("Bag evidence requires an object.") }; return fields
    }
    static func array(_ value: JSONValue?) throws -> [JSONValue] {
        guard case .array(let values) = value else { throw WorkflowPhaseError("Bag evidence requires an array.") }; return values
    }
    static func strings(_ value: JSONValue?) throws -> [String] {
        try array(value).map { guard case .string(let text) = $0 else { throw WorkflowPhaseError("Bag identity requires text.") }; return text }
    }
    static func json<T: Encodable>(_ value: T) throws -> JSONValue { try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(value)) }
    static func membership(_ ids: [String]) -> String {
        var bytes=Data("kwip.membership.v1\n".utf8)
        for id in ids.sorted(by: { Array($0.utf8).lexicographicallyPrecedes(Array($1.utf8)) }) {
            bytes.append(Data("\(id.utf8.count):".utf8));bytes.append(Data(id.utf8))
        }
        return WorkflowPhaseArtifacts.hash(bytes)
    }
    static func finiteFloat(_ value: JSONValue) throws -> Float {
        guard let n=value.doubleValue,n.isFinite,Float(n).isFinite,n == 0 || Float(n) != 0 else { throw WorkflowPhaseError("Mapped bag values must be finite float32 numbers.") };return Float(n)
    }
    private static func nested(_ cells: [JSONValue], shape: [Int]) throws -> JSONValue {
        let count=shape.reduce(1,*)
        try require(count == cells.count,"Objective columns do not fill their declared shape.")
        var index=0
        func build(_ depth:Int)->JSONValue {
            if depth == shape.count { defer {index += 1};return cells[index] }
            return .array((0..<shape[depth]).map {_ in build(depth+1)})
        }
        return build(0)
    }
    static func preparedEvidence(root: URL, parts: [String] = [], selection: WorkflowPreparedSelection,
                                 mapping: BagInputMapping, requirement: BagWorkflowInputRequirement) throws -> VerifiedBagPreparation {
        try selection.validate()
        let binary: String?, objective: PreparedObjectiveSelection?
        switch selection {case .legacy(let value):binary=value.targetColumn;objective=nil
        case .objective(let value):binary=nil;objective=value}
        try mapping.validate(sourceConfiguration:requirement.configuration,targetColumn:binary,objectiveMapping:objective?.targetMapping)
        let inventory=try WorkflowFailedGradientEvidence.preparedInventory(root:root,parts:parts,reference:selection.reference)
        let manifest=try object(JSONDecoder().decode(JSONValue.self,from:inventory.manifest))
        try require(manifest["schemaVersion"] == 1 && manifest["kind"] == .string("dataflowExecution") && manifest["status"] == .string("completed") && manifest["graphDigest"] == .string(selection.reference.graphDigest),"Bag preparation must be the selected completed source execution.")
        guard let registry=manifest["registryDigest"]?.stringValue,WorkflowArchiveCodec.isHash(registry) else {throw WorkflowPhaseError("Prepared registry pin is unavailable.")}
        var schemas:[[JSONValue]]=[]
        func partition(node:String,port:String,role:String) throws -> VerifiedBagPartition {
            let pinned=try WorkflowFailedGradientEvidence.preparedTable(root:root,prefix:inventory.prefix,entries:inventory.entries,node:node,port:port)
            let table=try object(JSONDecoder().decode(JSONValue.self,from:pinned.bytes))
            let columns=try array(table["columns"]),rows=try array(table["rows"]),lineage=try object(table["lineage"])
            try require(table["schemaVersion"] == 1 && table["kind"] == .string("table") && (1...4096).contains(rows.count) && rows.count <= 2_000_000 / max(1,requirement.embeddingDim),"Prepared bag table exceeds observation/output bounds.")
            schemas.append(columns)
            let ids=try strings(lineage["observationIDs"]),fitted=try strings(lineage["fittedStateDigests"])
            try require(ids.count == rows.count && Set(ids).count == ids.count && ids.allSatisfy {!$0.isEmpty && $0.utf8.count <= 1024} && lineage["role"] == .string(role),"Bag partition role or ordered row identity is invalid.")
            for key in ["sourceDigest","partitionDigest","dataVersion","membershipDigest"] {try require(lineage[key]?.stringValue.map(WorkflowArchiveCodec.isHash) == true,"Bag lineage digest is missing.")}
            try require(lineage["membershipDigest"] == .string(membership(ids)) && fitted.count <= 128 && fitted.allSatisfy(WorkflowArchiveCodec.isHash),"Bag membership or fitted state identity is invalid.")
            var positions:[String:Int]=[:],types:[String:String]=[:]
            try require((1...128).contains(columns.count),"Bag table column budget exceeded.")
            for (index,column) in columns.enumerated() {
                let c=try object(column)
                guard let name=c["name"]?.stringValue,let dtype=c["dtype"]?.stringValue,case .bool=c["nullable"],!name.isEmpty,name.unicodeScalars.count <= 200,positions[name] == nil else {throw WorkflowPhaseError("Bag column schema is invalid.")}
                positions[name]=index;types[name]=dtype
            }
            func position(_ name:String,dtype:String?=nil)throws->Int {
                guard let index=positions[name],dtype == nil || types[name] == dtype else {throw WorkflowPhaseError("Mapped bag column is missing or has the wrong type.")};return index
            }
            let indices=try mapping.indicesColumns.map {try position($0,dtype:"int64")}
            let weights=try (mapping.weightColumns ?? []).map {try position($0)}
            for name in mapping.weightColumns ?? [] {try require(["int64","float64"].contains(types[name] ?? ""),"Bag weights need numeric columns.")}
            let length=try mapping.lengthColumn.map {try position($0,dtype:"int64")}
            var prefix:[BagPreparedObservation]=[],labels=Set<Int>(),itemCounts:[Int]=[]
            let roles=objective?.targetMapping.roles ?? []
            for descriptor in roles {for name in descriptor.columns { _ = try position(name);try require(types[name] == "int64" || descriptor.dtype == .float32 && types[name] == "float64","Objective role column type differs.")}}
            let targetPosition=try binary.map {try position($0)}
            for rowValue in rows {
                let row=try array(rowValue);try require(row.count == columns.count,"Prepared bag row width differs.")
                let allIDs=try indices.map {index->Int64 in guard case .integer(let value)=row[index],value>=0,value<Int64(requirement.numEmbeddings) else {throw WorkflowPhaseError("Prepared item ID is outside the connected vocabulary.")};return value}
                let allWeights=try weights.map {try finiteFloat(row[$0])}
                let count:Int
                if let length {guard case .integer(let value)=row[length],value>=0,value<=Int64(requirement.configuration.maximumItemsPerBag) else {throw WorkflowPhaseError("Bag length must fit its whole observation.")};count=Int(value)} else {count=indices.count}
                itemCounts.append(count)
                var targets:[String:JSONValue]=[:]
                for descriptor in roles {
                    let values=try descriptor.columns.map {name->JSONValue in
                        let value=row[try position(name)]
                        if descriptor.dtype == .int64 {guard case .integer=value else {throw WorkflowPhaseError("Integer objective role must retain exact integers.")};return value}
                        return .number(Double(try finiteFloat(value)))
                    }
                    targets[descriptor.role]=try nested(values,shape:descriptor.sampleShape)
                }
                var label:Int?
                if let targetPosition {guard let value=row[targetPosition].integerValue,[0,1].contains(value) else {throw WorkflowPhaseError("Binary bag targets must be 0 or 1.")};label=value;labels.insert(value)}
                if prefix.count < 200 {prefix.append(.init(input:.init(indices:Array(allIDs.prefix(count)),perSampleWeights:mapping.weightColumns == nil ? nil:Array(allWeights.prefix(count))),targets:targets,binaryLabel:label))}
            }
            if binary != nil {try require(labels == [0,1],"Binary training requires both classes.")}
            var identity:[String:JSONValue]=["schemaVersion":1,"sourceConfiguration":.object(requirement.configuration.values),"mapping":mapping.value,"targetColumns":.array((binary.map {[$0]} ?? roles.flatMap(\.columns)).map(JSONValue.string)),"role":.string(role),"lineage":.object(lineage)]
            for key in ["sourceDigest","partitionDigest","dataVersion","membershipDigest","observationIDs","fittedStateDigests"] {identity[key]=lineage[key]}
            if let binary {identity["targetMapping"] = .object(["binaryTargetColumn":.string(binary),"requireBothBinaryClasses":.bool(true)])}
            else if let objective {identity["targetMapping"] = .object(["objectiveMapping":try json(objective.targetMapping),"objectiveRecipe":objective.objectiveRecipe])}
            return .init(count:rows.count,rowIDs:ids,tableSHA256:pinned.hash,lineage:lineage,prefix:prefix,sourceIdentity:identity,itemCounts:itemCounts)
        }
        let ref=selection.reference
        let train=try partition(node:ref.trainingNode,port:ref.trainingPort,role:"training"),val=try partition(node:ref.validationNode,port:ref.validationPort,role:"validation")
        try require(schemas[0] == schemas[1] && Set(train.rowIDs).isDisjoint(with:val.rowIDs),"Prepared bag partitions differ in schema or overlap.")
        for key in ["sourceDigest","partitionDigest","fittedStateDigests"] {try require(train.lineage[key] == val.lineage[key],"Prepared bag partitions do not share source, split and fitted states.")}
        for digest in try strings(train.lineage["fittedStateDigests"]) {
            var found=false
            for entry in inventory.entries where entry["kind"] as? String == "state" {
                guard let name=entry["path"] as? String,let size=entry["bytes"] as? Int,let hash=entry["sha256"] as? String,!name.contains("/"),(1...50_000_000).contains(size) else {throw WorkflowPhaseError("Prepared fitted state pin is invalid.")}
                let bytes=try DiagnosticFileIO.read(root:root,parts:inventory.prefix+[name],limit:size)
                try require(bytes.count == size && WorkflowPhaseArtifacts.hash(bytes) == hash,"Prepared fitted state changed.")
                let state=try object(JSONDecoder().decode(JSONValue.self,from:bytes))
                if state["contentDigest"] == .string(digest) {
                    let fit=try object(state["fitEvidence"])
                    try require(state["graphDigest"] == .string(ref.graphDigest) && state["partitionDigest"] == train.lineage["partitionDigest"] && fit["sourceDigest"] == train.lineage["sourceDigest"] && Set(try strings(fit["observationIDs"])).isSubset(of:Set(train.rowIDs)),"Fitted state includes heldout observations or a foreign source.");found=true
                }
            }
            try require(found,"Referenced fitted state is missing.")
        }
        let prep:[String:JSONValue]=["manifestSHA256":.string(ref.manifestSHA256),"graphDigest":.string(ref.graphDigest),"registryDigest":.string(registry),"trainingNode":.string(ref.trainingNode),"trainingPort":.string(ref.trainingPort),"validationNode":.string(ref.validationNode),"validationPort":.string(ref.validationPort),"trainingDataVersion":train.lineage["dataVersion"]!,"validationDataVersion":val.lineage["dataVersion"]!,"fittedStateDigests":train.lineage["fittedStateDigests"]!,"artifactDirectory":.string("preparation")]
        return .init(selection:selection,training:train,validation:val,preparationIdentity:prep)
    }
}
