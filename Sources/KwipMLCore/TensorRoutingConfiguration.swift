import Foundation

/// Strict routing rules shared by canvas declarations and independent lessons.
/// The leading observation direction is never joined, selected or contracted.
public struct TensorRoutingConfiguration:Equatable {
    public static let componentIDs:Set<String>=["split","chunk","gather","einsum"]
    public static let maximumElements=100_000_000
    public static let maximumOutputs=8
    public static let equationPresets:[String:String]=["multiply":"nab,nbc->nac","outer":"na,nb->nab","diagonal":"naa->na","reorder":"nab->nba"]
    public let componentID:String
    public let values:[String:JSONValue]
    public let dim:Int?
    public let equation:String?
    private let operandLabels:[[Character]]
    private let outputLabels:[Character]

    public init(componentID:String,values:[String:JSONValue])throws {
        guard Self.componentIDs.contains(componentID) else {throw GraphConfigurationError("Unknown tensor routing component.")}
        self.componentID=componentID
        var canonical:[String:JSONValue],selected:Int?,notation:String?,operands:[[Character]]=[],output:[Character]=[]
        func integer(_ key:String,_ defaultValue:Int,_ low:Int,_ high:Int)throws->Int {
            guard let number=(values[key] ?? .integer(Int64(defaultValue))).integerValue,(low...high).contains(number) else {throw GraphConfigurationError("\(key) must be a whole number from \(low) to \(high).")}
            return number
        }
        switch componentID {
        case "einsum":
            guard Set(values.keys).isSubset(of:["pattern","equation"]),let pattern=(values["pattern"] ?? .string("multiply")).stringValue,
                  pattern == "custom" || Self.equationPresets[pattern] != nil,
                  let supplied=(values["equation"] ?? .string(Self.equationPresets["multiply"]!)).stringValue,
                  (1...128).contains(supplied.unicodeScalars.count),supplied.unicodeScalars.allSatisfy({(32...126).contains(Int($0.value))}) else {
                throw GraphConfigurationError("Choose an arrangement preset or 1–128 printable ASCII characters of custom notation.")
            }
            notation=(Self.equationPresets[pattern] ?? supplied).replacingOccurrences(of:" ",with:"")
            let parsed=try Self.parseEquation(notation!);operands=parsed.0;output=parsed.1
            canonical=["pattern":.string(pattern),"equation":.string(notation!)]
        default:
            let allowed:Set<String>=componentID == "split" ? ["mode","dim","split_size","sections"] : componentID == "chunk" ? ["dim","chunks"]:["dim"]
            guard Set(values.keys).isSubset(of:allowed) else {throw GraphConfigurationError("Use only this routing block’s declared settings.")}
            selected=try integer("dim",-1,-8,7)
            guard selected != 0 else {throw GraphConfigurationError("Direction 0 belongs to observations; choose a sample direction.")}
            canonical=["dim":.integer(Int64(selected!))]
            if componentID == "chunk" {canonical["chunks"] = .integer(Int64(try integer("chunks",2,1,8)))}
            if componentID == "split" {
                guard let mode=(values["mode"] ?? .string("size")).stringValue,["size","sections"].contains(mode) else {throw GraphConfigurationError("Choose Size or Sections for the split arrangement.")}
                let size=try integer("split_size",2,1,Self.maximumElements)
                guard case .array(let raw)=(values["sections"] ?? .array([1,1])),(1...8).contains(raw.count) else {throw GraphConfigurationError("Supply 1–8 positive section sizes.")}
                let sections=raw.compactMap(\.integerValue)
                guard sections.count == raw.count,sections.allSatisfy({(1...Self.maximumElements).contains($0)}) else {throw GraphConfigurationError("Section sizes must be positive whole numbers no larger than 100 million.")}
                canonical["mode"] = .string(mode);canonical["split_size"] = .integer(Int64(mode == "size" ? size:2))
                canonical["sections"] = .array((mode == "sections" ? sections:[1,1]).map {.integer(Int64($0))})
            }
        }
        self.values=canonical;dim=selected;equation=notation;operandLabels=operands;outputLabels=output
    }

    private static func parseEquation(_ notation:String)throws->([[Character]],[Character]) {
        let sides=notation.components(separatedBy:"->")
        guard sides.count == 2 else {throw GraphConfigurationError("Custom notation needs exactly one arrow naming its output directions.")}
        let inputs=sides[0].components(separatedBy:",").map(Array.init),output=Array(sides[1])
        guard (1...2).contains(inputs.count) else {throw GraphConfigurationError("This block combines one or two source arrays.")}
        for labels in inputs+[output] {
            guard (2...8).contains(labels.count),labels.first == "n",labels.filter({$0 == "n"}).count == 1,
                  labels.allSatisfy({character in character.unicodeScalars.count == 1 && character.unicodeScalars.allSatisfy({(65...90).contains(Int($0.value)) || (97...122).contains(Int($0.value))})}) else {
                throw GraphConfigurationError("Use letter names for directions, with n once and first for observations and at least one sample direction. Ellipses and implicit outputs are unavailable.")
            }
        }
        guard Set(output).count == output.count,Set(output).isSubset(of:Set(inputs.flatMap {$0})) else {throw GraphConfigurationError("Output direction names must be unique and already occur in a source.")}
        return (inputs,output)
    }

    public var configuredInputs:[String] {
        switch componentID {case "gather":return ["source","indices"];case "einsum":return operandLabels.count == 1 ? ["a"]:["a","b"];default:return ["in"]}
    }
    public func sampleDirection(for layout:BatchedTensorLayout)throws->Int {
        guard let dim,(-layout.rank..<layout.rank).contains(dim) else {throw GraphConfigurationError("The selected direction is outside the incoming array’s rank.")}
        let normalized=dim<0 ? dim+layout.rank:dim
        guard normalized>0 else {throw GraphConfigurationError("This direction selects observations. Choose a direction inside each sample.")}
        return normalized-1
    }
    public func resolved(layouts:[BatchedTensorLayout])throws->[String:BatchedTensorLayout] {
        guard layouts.count == configuredInputs.count,let first=layouts.first else {throw GraphConfigurationError("Connect each configured input exactly once.")}
        switch componentID {
        case "split","chunk":
            let axis=try sampleDirection(for:first),extent=first.sample.dimensions[axis]
            let sizes:[Int]
            if componentID == "split",values["mode"]?.stringValue == "sections" {
                guard case .array(let raw)=values["sections"] else {throw GraphConfigurationError("Section sizes are unavailable.")}
                sizes=raw.compactMap(\.integerValue)
                guard sizes.reduce(0,+) == extent else {throw GraphConfigurationError("Section sizes must add up to the selected incoming direction’s size \(extent).")}
            } else {
                let requested=componentID == "split" ? values["split_size"]!.integerValue!:values["chunks"]!.integerValue!
                let step=componentID == "split" ? requested:(extent+requested-1)/requested
                let count=(extent+step-1)/step
                guard count<=Self.maximumOutputs else {throw GraphConfigurationError("This split produces \(count) parts. Use at most eight actual parts, or chain another split.")}
                sizes=(0..<count).map {min(step,extent-$0*step)}
            }
            return try Dictionary(uniqueKeysWithValues:sizes.enumerated().map {index,size in
                var dimensions=first.sample.dimensions;dimensions[axis]=size
                return ("part\(index)",try BatchedTensorLayout(sample:dimensions))
            })
        case "gather":
            let second=layouts[1],axis=try sampleDirection(for:first)
            guard first.rank == second.rank,first.sample.dimensions.indices.allSatisfy({$0 == axis || second.sample.dimensions[$0]<=first.sample.dimensions[$0]}) else {throw GraphConfigurationError("Position arrays need the same rank; every other sample direction must fit the source. Observations cannot broadcast.")}
            return ["out":second]
        default:
            var extents:[Character:Int]=[:]
            for (labels,layout) in zip(operandLabels,layouts) {
                guard labels.count == layout.rank else {throw GraphConfigurationError("Give each incoming direction one notation name, including observations.")}
                var local:[Character:Int]=[:]
                for (label,size) in zip(labels.dropFirst(),layout.sample.dimensions) {
                    if let seen=local[label],seen != size {throw GraphConfigurationError("Repeated direction names inside one source need equal sizes for a diagonal.")}
                    local[label]=size
                }
                for (label,size) in local {
                    if let seen=extents[label],seen != size,seen != 1,size != 1 {throw GraphConfigurationError("Shared sample direction \(label) must match or have size one.")}
                    extents[label]=max(extents[label] ?? 1,size)
                }
            }
            var work=1
            for size in extents.values {
                guard work<=Self.maximumElements/size else {throw GraphConfigurationError("This arrangement exceeds 100 million coordinate terms per observation. Reduce the incoming dimensions.")}
                work*=size
            }
            return ["out":try .init(sample:outputLabels.dropFirst().map {extents[$0]!})]
        }
    }
    public func resolved(inputs:[TypedTensorContract])throws->[String:TypedTensorContract] {
        guard inputs.count == configuredInputs.count else {throw GraphConfigurationError("Connect each configured input exactly once.")}
        for (index,input) in inputs.enumerated() {
            let expected:ContractDType=componentID == "gather" && index == 1 ? .int64:.float32
            guard input.dtype == expected else {throw GraphConfigurationError(expected == .int64 ? "Positions must be whole-number int64 arrays; they are not learned values.":"Tensor routing measurements require 32-bit decimal arrays.")}
        }
        return try resolved(layouts:inputs.map(\.layout)).mapValues {.init(layout:$0,dtype:.float32)}
    }
    public func outputPorts(layouts:[BatchedTensorLayout])throws->[String] {
        let outputs=try resolved(layouts:layouts)
        return componentID == "split" || componentID == "chunk" ? (0..<outputs.count).map {"part\($0)"}:["out"]
    }
}
