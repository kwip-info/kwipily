import Foundation

/// Canonical training-objective settings; never a model-forward tensor operation.
public struct LossConfiguration: Equatable {
    public static let componentIDs: Set<String> = ["loss_cross_entropy", "loss_bce_with_logits", "loss_mse", "loss_l1", "loss_nll", "loss_kl_div", "loss_ctc", "loss_triplet_margin"]
    public static let maximumRecipeBytes = 32768
    public static let scalarMaximum = 3.4028234663852886e38
    public static let scalarMinimumPositive = 1.401298464324817e-45
    public let componentID: String
    public let values: [String: JSONValue]
    public var recipeValue: JSONValue { Self.envelope(componentID:componentID,values:values) }

    public init(componentID: String, values supplied: [String: JSONValue]) throws {
        guard Self.componentIDs.contains(componentID) else {throw GraphConfigurationError("Choose one of the supported loss objectives.")}
        var values:[String:JSONValue] = ["reduction":.string("mean")]
        switch componentID {
        case "loss_cross_entropy":
            values.merge(["target_mode":.string("class_indices"),"class_axis":-1,"class_weights":.array([]),"ignore_index":-100,"label_smoothing":.number(0)]) {_,new in new}
        case "loss_bce_with_logits":
            for prefix in ["weight","pos_weight"] {values[prefix+"_mode"] = .string("none");values[prefix+"_shape"] = .array([]);values[prefix+"_values"] = .array([])}
        case "loss_mse","loss_l1":values["target_broadcast"] = false
        case "loss_nll":values.merge(["class_axis":-1,"class_weights":.array([]),"ignore_index":-100]) {_,new in new}
        case "loss_kl_div":values["log_target"] = false
        case "loss_ctc":values.merge(["blank":0,"zero_infinity":false,"target_layout":.string("padded")]) {_,new in new}
        default:values.merge(["margin":.number(1),"p":.number(2),"eps":.number(1e-6),"swap":false]) {_,new in new}
        }
        guard Set(supplied.keys).isSubset(of:Set(values.keys)) else {throw GraphConfigurationError("Use only this objective's declared controls; targets and learned state are bound separately.")}
        values.merge(supplied) {_,new in new}
        for key in values.keys {
            let value=values[key]!
            switch key {
            case "reduction":
                try Self.choice(value,key:key,allowed:componentID == "loss_kl_div" ? ["none","mean","sum","batchmean"]:["none","mean","sum"])
            case "target_mode":try Self.choice(value,key:key,allowed:["class_indices","probabilities"])
            case "target_layout":try Self.choice(value,key:key,allowed:["padded","packed"])
            case "weight_mode","pos_weight_mode":try Self.choice(value,key:key,allowed:["none","configured","mapped"])
            case "target_broadcast","log_target","zero_infinity","swap":
                guard value.boolValue != nil else {throw GraphConfigurationError("\(key) must be a Boolean switch.")}
            case "class_axis","ignore_index","blank":
                let bounds = key == "class_axis" ? -7...7:key == "blank" ? 0...511:Int(Int32.min)...Int(Int32.max)
                let number=try Self.integer(value,key:key,bounds:bounds)
                guard key != "class_axis" || number != 0 else {throw GraphConfigurationError("Class direction cannot be the observation group direction, 0.")}
                values[key] = .integer(Int64(number))
            case "weight_shape","pos_weight_shape":
                guard case .array(let items)=value,items.count<=7 else {throw GraphConfigurationError("Weight shape requires at most seven sample directions.")}
                let shape=try items.map {try Self.integer($0,key:key,bounds:1...4096)}
                _ = try Self.elements(shape)
                values[key] = .array(shape.map {.integer(Int64($0))})
            case "class_weights","weight_values","pos_weight_values":
                guard case .array(let items)=value,items.count<=(key == "class_weights" ? 512:4096) else {throw GraphConfigurationError("Weight values exceed this objective's declared list limit.")}
                values[key] = .array(try items.map {.number(try Self.number($0,key:key,minimum:0,maximum:Self.scalarMaximum))})
            default:
                let minimum=["margin","p"].contains(key) ? Self.scalarMinimumPositive:0
                let maximum=key == "p" ? 16:key == "label_smoothing" ? 1:Self.scalarMaximum
                values[key] = .number(try Self.number(value,key:key,minimum:minimum,maximum:maximum))
            }
        }
        if componentID == "loss_cross_entropy",values["target_mode"] == .string("probabilities") {
            guard values["ignore_index"] == .integer(-100) else {throw GraphConfigurationError("Class shares cannot use an ignored class index. Keep Ignore index at −100.")}
        }
        if componentID == "loss_bce_with_logits" {
            for prefix in ["weight","pos_weight"] {
                guard case .array(let shape)=values[prefix+"_shape"],case .array(let items)=values[prefix+"_values"] else {throw GraphConfigurationError("Declare weight shapes and numeric values.")}
                let count=try Self.elements(shape.map {$0.integerValue!})
                guard (items.isEmpty || items.count == count),values[prefix+"_mode"] != .string("configured") || items.count == count else {throw GraphConfigurationError("Configured weight values must fill their sample shape exactly. Shape [] requires one scalar value.")}
            }
        }
        try Self.checkSize(Self.envelope(componentID:componentID,values:values))
        self.componentID=componentID;self.values=values
    }

    public static func decodeRecipe(_ value: JSONValue) throws -> LossConfiguration {
        guard case .object(let envelope)=value,Set(envelope.keys) == ["schemaVersion","componentID","configurationVersion","config"],
              envelope["schemaVersion"] == .integer(1),envelope["configurationVersion"] == .integer(1),
              let id=envelope["componentID"]?.stringValue,case .object(let config)=envelope["config"] else {throw GraphConfigurationError("Objective recipes require version 1, component identity and declared configuration only.")}
        try checkSize(value)
        return try .init(componentID:id,values:config)
    }

    /// Structural table bindings are checked here; values and prediction geometry are checked at execution.
    public func validate(targetMapping: LossTargetMapping) throws {
        try targetMapping.validate()
        guard targetMapping.componentID == componentID else {throw GraphConfigurationError("Bind targets to the same selected objective.")}
        if targetMapping.schemaVersion==2 {
            guard componentID=="loss_cross_entropy",values==((try? LossConfiguration(componentID:"loss_cross_entropy",values:[:]))?.values) else {throw GraphConfigurationError("Next-token training requires unweighted, unsmoothed valid-token mean cross entropy with ignore index −100.")}
        }
        let roles=Dictionary(uniqueKeysWithValues:targetMapping.roles.map {($0.role,$0)})
        if componentID == "loss_cross_entropy" {
            let expected:ContractDType = values["target_mode"] == .string("probabilities") ? .float32:.int64
            guard roles["target"]?.dtype == expected else {throw GraphConfigurationError("Target columns must use the selected class-index or class-share format.")}
        }
        if componentID == "loss_bce_with_logits" {
            for prefix in ["weight","pos_weight"] {
                let mapped=values[prefix+"_mode"] == .string("mapped")
                guard (roles[prefix] != nil) == mapped else {throw GraphConfigurationError("Bind the \(prefix) target role exactly when its mode uses mapped data.")}
            }
        }
    }

    private static func envelope(componentID:String,values:[String:JSONValue])->JSONValue {.object(["schemaVersion":1,"componentID":.string(componentID),"configurationVersion":1,"config":.object(values)])}
    private static func checkSize(_ value:JSONValue)throws {
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        guard value.isFiniteJSON,let data=try? encoder.encode(value),data.count<=maximumRecipeBytes else {throw GraphConfigurationError("Canonical objective recipes must fit within 32768 UTF8 bytes and contain finite values.")}
        // Python's canonical float serializer adds ".0" where Swift emits an integral real
        // without a decimal/exponent. Reserve those bytes before the transport boundary.
        func realSpellingAllowance(_ value:JSONValue)throws->Int {
            switch value {
            case .number:
                let bytes=try encoder.encode(value)
                let spelling=String(decoding:bytes,as:UTF8.self)
                return spelling.contains(".") || spelling.contains("e") || spelling.contains("E") ? 0:2
            case .array(let items):return try items.reduce(0) {try $0+realSpellingAllowance($1)}
            case .object(let entries):return try entries.values.reduce(0) {try $0+realSpellingAllowance($1)}
            default:return 0
            }
        }
        guard data.count + (try realSpellingAllowance(value))<=maximumRecipeBytes else {throw GraphConfigurationError("Objective numeric values exceed the canonical 32768-byte transport budget. Use fewer configured weights or map weights from the data.")}
    }
    private static func choice(_ value:JSONValue,key:String,allowed:[String])throws {
        guard let choice=value.stringValue,allowed.contains(choice) else {throw GraphConfigurationError("Choose a supported \(key) value: \(allowed.joined(separator:", ")).")}
    }
    private static func integer(_ value:JSONValue,key:String,bounds:ClosedRange<Int>)throws->Int {
        guard let number=value.doubleValue,number.isFinite,number.rounded(.towardZero)==number,Double(bounds.lowerBound)<=number,number<=Double(bounds.upperBound) else {throw GraphConfigurationError("\(key) must be a whole number from \(bounds.lowerBound) to \(bounds.upperBound).")}
        return Int(number)
    }
    private static func number(_ value:JSONValue,key:String,minimum:Double,maximum:Double)throws->Double {
        guard let number=value.doubleValue,number.isFinite,(minimum...maximum).contains(number) else {throw GraphConfigurationError("\(key) must be a finite number in its declared range.")}
        let rounded=Float(number)
        guard rounded.isFinite,number == 0 || rounded != 0 else {throw GraphConfigurationError("\(key) is too small or large to represent as a float32 value.")}
        return number
    }
    private static func elements(_ shape:[Int])throws->Int {
        var count=1
        for dimension in shape {
            guard count<=100_000_000/dimension else {throw GraphConfigurationError("Weight geometry exceeds the 100M element budget.")}
            count *= dimension
        }
        return count
    }
}
