import Foundation

/// Source-independent preparation/mapping contract. Names are semantic API names,
/// not node labels. Image/audio representations are reserved, never implicit reads.
public struct NamedInputPlan: Codable, Equatable {
    public struct Tensor: Codable, Equatable, Identifiable {
        public var name:String
        public var nodeID:String?
        public var representation:String
        public var dtype:String
        public var sampleAxes:[String]?
        public var sampleShape:[Int]
        public var columns:[String]
        public var lengthPolicy:String?
        public var paddingValue:Int?
        public var vocabularySize:Int?
        public var id:String {name}
        public init(name:String,nodeID:String?=nil,representation:String="numeric",dtype:String="float32",sampleShape:[Int],columns:[String],lengthPolicy:String?=nil,paddingValue:Int?=nil,vocabularySize:Int?=nil,sampleAxes:[String]?=nil) {
            self.name=name;self.nodeID=nodeID;self.representation=representation;self.dtype=dtype;self.sampleShape=sampleShape;self.columns=columns;self.lengthPolicy=lengthPolicy;self.paddingValue=paddingValue;self.vocabularySize=vocabularySize;self.sampleAxes=sampleAxes
        }
        public var elementCount:Int {sampleShape.reduce(1,*)}
    }
    public struct SequenceObjective:Codable,Equatable {
        public var schemaVersion:Int=1
        public var inputName:String
        public var targetName:String="target"
        public var lengthColumn:String
        public var maskColumn:String
        public init(inputName:String,lengthColumn:String="valid_length",maskColumn:String="loss_mask"){self.inputName=inputName;self.lengthColumn=lengthColumn;self.maskColumn=maskColumn}
    }
    public var sequenceObjective:SequenceObjective?=nil
    public var schemaVersion:Int=1
    public var inputs:[Tensor]
    public var roles:[Tensor]
    public var maximumBatchBytes:Int
    public init(inputs:[Tensor],roles:[Tensor]=[],maximumBatchBytes:Int=64*1024*1024) {self.inputs=inputs;self.roles=roles;self.maximumBatchBytes=maximumBatchBytes}
    public init(values:[String:JSONValue])throws {
        guard Set(values.keys)==Set(["schemaVersion","inputs","roles","maximumBatchBytes"]).union(values["schemaVersion"] == 2 ? ["sequenceObjective"]:[]) else {throw DataFlowContractError("Unknown or missing named input plan field.")}
        for group in ["inputs","roles"] {
            guard case .array(let entries)=values[group] else {throw DataFlowContractError("Named mappings require input/role lists.")}
            for entry in entries {
                guard case .object(let row)=entry else {throw DataFlowContractError("Each named mapping must be an object.")}
                let required:Set<String>=["name","representation","dtype","sampleShape","columns"]
                guard required.isSubset(of:Set(row.keys)),Set(row.keys).isSubset(of:required.union(["nodeID","lengthPolicy","paddingValue","vocabularySize"]).union(values["schemaVersion"] == 3 ? ["sampleAxes"]:[])) else {throw DataFlowContractError("Unknown named tensor mapping field.")}
            }
        }
        if let raw=values["sequenceObjective"] {guard case .object(let fields)=raw,Set(fields.keys)==["schemaVersion","inputName","targetName","lengthColumn","maskColumn"] else {throw DataFlowContractError("Unknown sequence objective mapping field.")}}
        self=try JSONDecoder().decode(Self.self,from:JSONEncoder().encode(JSONValue.object(values)))
        try validate()
    }
    public var values:[String:JSONValue] {guard let value=try? JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(self)),case .object(let object)=value else {return [:]};return object}
    public func validate(executable:Bool=true)throws {
        guard (([1,3].contains(schemaVersion) && sequenceObjective==nil) || (schemaVersion==2 && sequenceObjective != nil)),(1...8).contains(inputs.count),roles.count<=8,(1...268435456).contains(maximumBatchBytes) else {throw DataFlowContractError("Named inputs require a supported version, 1–8 inputs and a bounded batch allowance.")}
        for (isInput,entries) in [(true,inputs),(false,roles)] {
            guard Set(entries.map(\.name)).count==entries.count else {throw DataFlowContractError("Input and target names must be unique within their group.")}
            if isInput {guard Set(entries.compactMap(\.nodeID)).count==entries.count else {throw DataFlowContractError("Each named input must own a distinct model boundary.")}}
            for e in entries {
                guard e.name.range(of:"^[A-Za-z][A-Za-z0-9_]{0,63}$",options:.regularExpression) != nil,
                      ["numeric","tokens","prepared","image","audio"].contains(e.representation),["float32","int64"].contains(e.dtype),
                      ((isInput ? 1:0)...7).contains(e.sampleShape.count),e.sampleShape.allSatisfy({(1...(e.representation=="prepared" ? 1_048_576:4096)).contains($0)}),
                      (1...128).contains(e.columns.count),Set(e.columns).count==e.columns.count,e.columns.allSatisfy({!$0.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty && $0.count<=200 && !$0.contains("\0")}) else {throw DataFlowContractError("Named tensor shape, dtype or columns are invalid.")}
                var count=1
                for d in e.sampleShape {guard count<=1048576/d else {throw DataFlowContractError("A tensor sample exceeds its element bound.")};count *= d}
                if isInput {guard let id=e.nodeID,!id.isEmpty,id.count<=128,!id.contains(".") else {throw DataFlowContractError("Name an existing model input boundary.")}}
                else if e.nodeID != nil {throw DataFlowContractError("Target roles cannot own model input nodes.")}
                if executable && ["image","audio"].contains(e.representation) {throw DataFlowContractError("Image/audio tensor adapters are not available in this checkpoint.")}
                if e.representation=="prepared" {
                    guard schemaVersion==3,e.columns.count==1,let axes=e.sampleAxes,axes.count==e.sampleShape.count,Set(axes).count==axes.count,
                          axes.allSatisfy({["samples","channels","height","width","features","tokens","frequency","time"].contains($0)}),e.lengthPolicy==nil,e.paddingValue==nil,e.vocabularySize==nil else {throw DataFlowContractError("Prepared tensors require exact named axes and one captured tensor column in mapping v3.")}
                } else if e.sampleAxes != nil {throw DataFlowContractError("Named media axes belong only to prepared tensor mappings.")}
                if e.representation=="numeric" {guard e.columns.count==count,e.lengthPolicy==nil,e.paddingValue==nil,e.vocabularySize==nil else {throw DataFlowContractError("Numeric columns must fill the exact declared shape.")}}
                if e.representation=="tokens" {guard e.dtype=="int64",e.columns.count==1,e.sampleShape.count==1,
                    ["error","pad","truncate","padTruncate"].contains(e.lengthPolicy ?? ""),let vocabulary=e.vocabularySize,(1...65536).contains(vocabulary),let padding=e.paddingValue,(0..<vocabulary).contains(padding) else {throw DataFlowContractError("Tokens require an integer sequence, explicit length policy and valid vocabulary/padding IDs.")}}
            }
        }
        if let seq=sequenceObjective {
            guard seq.schemaVersion==1,seq.targetName=="target",roles.count==1,roles[0].name=="target",let x=inputs.first(where:{$0.name==seq.inputName}) else {throw DataFlowContractError("Sequence objective binds one existing token input and the target role.")}
            let y=roles[0]
            guard [x,y].allSatisfy({$0.representation=="tokens" && $0.lengthPolicy=="error"}),x.sampleShape==y.sampleShape,x.vocabularySize==y.vocabularySize,x.paddingValue==y.paddingValue,(2...512).contains(x.vocabularySize ?? 0) else {throw DataFlowContractError("Sequence input/target require matching exact shape, vocabulary and padding, with 2–512 classes.")}
            let columns=Set((inputs+roles).flatMap(\.columns))
            guard !seq.lengthColumn.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,[seq.lengthColumn,seq.maskColumn].allSatisfy({$0.count<=200 && !$0.contains("\0") && !columns.contains($0)}),seq.lengthColumn != seq.maskColumn else {throw DataFlowContractError("Choose separate valid-length and optional loss-mask metadata columns.")}
        }
        guard Set(inputs.flatMap(\.columns)).isDisjoint(with:roles.flatMap(\.columns)) else {throw DataFlowContractError("Outcome columns cannot also be model input features.")}
    }
}
