import Foundation

extension JSONValue: ExpressibleByIntegerLiteral, ExpressibleByFloatLiteral,
  ExpressibleByBooleanLiteral
{
  public init(integerLiteral value: Int64) { self = .integer(value) }
  public init(floatLiteral value: Double) { self = .number(value) }
  public init(booleanLiteral value: Bool) { self = .bool(value) }
  public var stringValue: String? {
    if case .string(let value) = self { return value }
    return nil
  }
  public var doubleValue: Double? {
    switch self {
    case .integer(let value): return Double(value)
    case .number(let value): return value.isFinite ? value : nil
    default: return nil
    }
  }
  public var integerValue: Int? {
    switch self {
    case .integer(let value): return Int(exactly: value)
    case .number(let value): return Int(exactly: value)
    default: return nil
    }
  }
  public var boolValue: Bool? {
    if case .bool(let value) = self { return value }
    return nil
  }
  public var isFiniteJSON: Bool {
    switch self {
    case .number(let value): return value.isFinite
    case .array(let values): return values.allSatisfy(\.isFiniteJSON)
    case .object(let values): return values.values.allSatisfy(\.isFiniteJSON)
    default: return true
    }
  }
}

public enum ComponentFieldKind: String, Codable { case integer, number, boolean, integerList, numberList, choice, text }

extension ComponentField {
  public var isValidDefinition: Bool {
    guard !key.isEmpty, kind == .choice || choices == nil, !allowsNull || kind == .integer || kind == .number || kind == .integerList || kind == .numberList else { return false }
    if let enabledValue {
      guard allowsNull, enabledValue != .null, accepts(enabledValue) else { return false }
    }
    switch kind {
    case .numberList:
      guard let min,let max,min.isFinite,max.isFinite,min<=max,
            let minItems,let maxItems,minItems>=0,minItems<=maxItems,maxItems<=4096 else {return false}
      return accepts(defaultValue)
    case .text:
      guard min == nil,max == nil,let minItems,let maxItems,(1...4096).contains(minItems),minItems<=maxItems,maxItems<=4096,!allowsNull else {return false}
      return accepts(defaultValue)
    case .choice:
      guard min == nil, max == nil, minItems == nil, maxItems == nil, !allowsNull,
            hasValidChoices else { return false }
      return accepts(defaultValue)
    case .number:
      guard minItems == nil, maxItems == nil, let min, let max, min.isFinite, max.isFinite, min <= max else { return false }
      return accepts(defaultValue)
    case .boolean: return min == nil && max == nil && minItems == nil && maxItems == nil && defaultValue.boolValue != nil
    case .integer, .integerList:
      if kind == .integerList {
        guard let minItems,let maxItems,minItems>=0,maxItems>=minItems,maxItems<=4096 else {return false}
      } else if minItems != nil || maxItems != nil {return false}
      guard let min, let max, min.isFinite, max.isFinite, min <= max, Int(exactly: min) != nil,
        Int(exactly: max) != nil
      else { return false }
      return accepts(defaultValue)
    }
  }
  private var hasValidChoices: Bool {
    guard let choices, (1...64).contains(choices.count), Set(choices.map(\.value)).count == choices.count else { return false }
    return choices.allSatisfy {
      !$0.value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && $0.value.unicodeScalars.count <= 128
      && !$0.title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && $0.title.unicodeScalars.count <= 128
    }
  }
  /// A concrete in-range starting value when an optional setting is enabled.
  public var valueWhenEnabled:JSONValue {
    guard isValidDefinition else {return .null}
    if let enabledValue {return enabledValue}
    if defaultValue != .null {return defaultValue}
    guard let min,let max else {return .null}
    // Zero is natural for optional indices. Strictly positive real settings
    // start at one (bounded to the domain), not at a near-zero numeric limit.
    let preferred:Double = kind == .number && min>0 ? 1 : 0
    let value=Swift.min(max,Swift.max(min,preferred))
    if kind == .numberList,let count=minItems {return .array(Array(repeating:.number(Swift.min(max,Swift.max(min,1))),count:count))}
    if kind == .integerList,let count=minItems,let integer=Int64(exactly:value) {return .array(Array(repeating:.integer(integer),count:count))}
    if kind == .integer,let integer=Int64(exactly:value) {return .integer(integer)}
    return .number(value)
  }
  public func accepts(_ value: JSONValue) -> Bool {
    if value == .null {return allowsNull && (kind == .integer || kind == .number || kind == .integerList || kind == .numberList)}
    switch kind {
    case .numberList:
      guard case .array(let items)=value,let minItems,let maxItems,
            minItems>=0,maxItems<=4096,minItems<=maxItems,(minItems...maxItems).contains(items.count),
            let min,let max,min.isFinite,max.isFinite,min<=max else {return false}
      return items.allSatisfy {item in item.doubleValue.map {$0>=min && $0<=max} == true}
    case .text:
      guard let text=value.stringValue,let minimum=minItems,let maximum=maxItems,(1...4096).contains(minimum),minimum<=maximum,maximum<=4096 else {return false}
      return (minimum...maximum).contains(text.unicodeScalars.count)
    case .choice:
      guard hasValidChoices, let selected = value.stringValue else { return false }
      return choices!.contains { $0.value == selected }
    case .integerList:
      guard case .array(let items)=value,let minItems,let maxItems,
            minItems>=0,maxItems<=4096,minItems<=maxItems,(minItems...maxItems).contains(items.count),
            let min,let max else {return false}
      return items.allSatisfy {item in
        guard let number=item.integerValue else {return false}
        return Double(number)>=min && Double(number)<=max
      }
    case .number:
      guard let number = value.doubleValue, let min, let max, min.isFinite, max.isFinite else { return false }
      return number >= min && number <= max
    case .boolean: return value.boolValue != nil
    case .integer:
      guard let number = value.integerValue, let min, let max else { return false }
      return Double(number) >= min && Double(number) <= max
    }
  }
}

public struct GraphConfigurationError: Error, LocalizedError {
  public let message: String
  public init(_ message: String) { self.message = message }
  public var errorDescription: String? { message }
}

public enum GraphMigration {
  public static func effectiveConfiguration(
    of node: GraphNode, schemaVersion: Int, registry: ComponentRegistry
  ) throws -> [String: JSONValue] {
    guard [1, 2].contains(schemaVersion), let definition = registry.component(type: node.type)
    else { throw GraphConfigurationError("Unknown graph version or component type: \(node.type).") }
    guard !["gelu", "sigmoid", "silu", "leaky_relu", "elu"].contains(node.type) || schemaVersion == 2 else {
      throw GraphConfigurationError("\(definition.title) requires graph configuration schema 2.")
    }
    if node.type == "input",node.config["shape"] != nil {
      guard schemaVersion==2 else {throw GraphConfigurationError("Shaped floating input requires graph schema2.")}
      _=try FloatingInputConfiguration(node.config);return node.config
    }
    if node.type == "mixture_experts" {
      guard schemaVersion==2 else {throw GraphConfigurationError("Mixture of experts requires schema2.")}
      return try MixtureExpertsConfiguration(values:node.config,registry:registry).values
    }
    if node.type == "output" {
      let value = try ModelOutputConfiguration(node.config)
      return node.config.isEmpty ? [:] : value.values
    }
    if AttentionConfiguration.componentIDs.contains(node.type) {
      guard schemaVersion == 2 else {throw GraphConfigurationError("Attention components require graph configuration schema 2.")}
      return try AttentionConfiguration(componentID:node.type,values:node.config).values
    }
    if RecurrentConfiguration.componentIDs.contains(node.type) {
      guard schemaVersion == 2 else {throw GraphConfigurationError("Memory components require graph configuration schema 2.")}
      return try RecurrentConfiguration(componentID:node.type,values:node.config).values
    }
    if StatefulNormalizationConfiguration.componentIDs.contains(node.type) {
      guard schemaVersion == 2 else { throw GraphConfigurationError("Channel normalization requires graph configuration schema 2.") }
      let canonical = try StatefulNormalizationConfiguration(componentID: node.type, values: node.config).values
      guard Set(canonical.keys) == Set(definition.fields.map(\.key)), definition.fields.allSatisfy({ $0.isValidDefinition && canonical[$0.key].map($0.accepts) == true }) else { throw GraphConfigurationError("Channel normalization differs from its reviewed field template.") }
      return canonical
    }
    if SpatialPatchConfiguration.componentIDs.contains(node.type) {
      guard schemaVersion == 2 else { throw GraphConfigurationError("Spatial patches require graph configuration schema 2.") }
      let canonical = try SpatialPatchConfiguration(componentID: node.type, values: node.config).values
      guard Set(canonical.keys) == Set(definition.fields.map(\.key)), definition.fields.allSatisfy({ $0.isValidDefinition && canonical[$0.key].map($0.accepts) == true }) else { throw GraphConfigurationError("Patch geometry differs from its reviewed field template.") }
      return canonical
    }
    if TensorRoutingConfiguration.componentIDs.contains(node.type) {
      guard schemaVersion == 2 else {throw GraphConfigurationError("Tensor routing requires graph configuration schema 2.")}
      let canonical=try TensorRoutingConfiguration(componentID:node.type,values:node.config).values
      guard Set(canonical.keys) == Set(definition.fields.map(\.key)),definition.fields.allSatisfy({$0.isValidDefinition && canonical[$0.key].map($0.accepts) == true}) else {throw GraphConfigurationError("Routing differs from its reviewed field template.")}
      return canonical
    }
    if RemainingActivationConfiguration.componentIDs.contains(node.type) {
      guard schemaVersion == 2 else {
        throw GraphConfigurationError("\(definition.title) requires graph configuration schema 2.")
      }
      let canonical = try RemainingActivationConfiguration(componentID: node.type, values: node.config).values
      guard Set(canonical.keys) == Set(definition.fields.map(\.key)), definition.fields.allSatisfy({ $0.isValidDefinition && canonical[$0.key].map($0.accepts) == true }) else {
        throw GraphConfigurationError("Activation configuration differs from its reviewed field template.")
      }
      return canonical
    }
    if node.type == "matmul" {
      guard schemaVersion == 2,node.config.isEmpty else {
        throw GraphConfigurationError("Matrix multiplication requires schema 2 and no configuration fields; reshape or transpose with explicit nodes.")
      }
    }
    if TensorCompositionConfiguration.componentIDs.contains(node.type) {
      guard ["concat", "add"].contains(node.type) || schemaVersion == 2 else {
        throw GraphConfigurationError("Stack and Permute require graph configuration schema 2.")
      }
      let canonical = try TensorCompositionConfiguration(componentID: node.type, values: node.config).values
      guard Set(canonical.keys) == Set(definition.fields.map(\.key)), definition.fields.allSatisfy({ $0.isValidDefinition && canonical[$0.key].map($0.accepts) == true }) else {
        throw GraphConfigurationError("Composition configuration differs from the reviewed field template.")
      }
      return canonical
    }
    let fields = definition.fields.filter {
      !(schemaVersion == 1 && node.type == "linear" && $0.key == "bias")
    }
    guard Set(node.config.keys) == Set(fields.map(\.key)) else {
      throw GraphConfigurationError(
        "\(definition.title) configuration must contain exactly: \(fields.map(\.key).joined(separator:", "))."
      )
    }
    for field in fields {
      guard field.isValidDefinition, let value = node.config[field.key], field.accepts(value) else {
        throw GraphConfigurationError(
          "\(field.title) must match its declared \(field.kind.rawValue) type and bounds.")
      }
    }
    if ConfigurableActivationConfiguration.componentIDs.contains(node.type) {
      return try ConfigurableActivationConfiguration(componentID: node.type, values: node.config).values
    }
    if node.type == "max_pool3d" {
      guard schemaVersion == 2 else { throw GraphConfigurationError("MaxPool3d requires graph configuration schema 2.") }
      return try MaxPool3dConfiguration(values: node.config).values
    }
    if node.type == "max_pool2d" {
      guard schemaVersion == 2 else { throw GraphConfigurationError("MaxPool2d requires graph configuration schema 2.") }
      return try MaxPool2dConfiguration(values: node.config).values
    }
    if node.type == "avg_pool1d" {
      guard schemaVersion == 2 else { throw GraphConfigurationError("AvgPool1d requires graph configuration schema 2.") }
      return try AvgPool1dConfiguration(values: node.config).values
    }
    if node.type == "avg_pool2d" || AdaptivePoolingConfiguration.componentIDs.contains(node.type) {
      guard schemaVersion == 2 else { throw GraphConfigurationError("Pooling requires graph configuration schema 2.") }
      if node.type == "avg_pool2d" { return try AvgPool2dConfiguration(values: node.config).values }
      return try AdaptivePoolingConfiguration(componentID: node.type, values: node.config).values
    }
    if node.type == "max_pool1d" {
      guard schemaVersion == 2 else { throw GraphConfigurationError("MaxPool1d requires graph configuration schema 2.") }
      return try MaxPool1dConfiguration(values: node.config).values
    }
    if node.type == "depthwise_conv2d" {
      guard schemaVersion == 2 else { throw GraphConfigurationError("DepthwiseConv2d requires graph configuration schema 2.") }
      return try DepthwiseConv2dConfiguration(values: node.config).values
    }
    if node.type == "conv_transpose3d" {
      guard schemaVersion == 2 else { throw GraphConfigurationError("ConvTranspose3d requires graph configuration schema 2.") }
      return try ConvTranspose3dConfiguration(values: node.config).values
    }
    if node.type == "conv_transpose2d" {
      guard schemaVersion == 2 else { throw GraphConfigurationError("ConvTranspose2d requires graph configuration schema 2.") }
      return try ConvTranspose2dConfiguration(values: node.config).values
    }
    if node.type == "conv_transpose1d" {
      guard schemaVersion == 2 else { throw GraphConfigurationError("ConvTranspose1d requires graph configuration schema 2.") }
      return try ConvTranspose1dConfiguration(values: node.config).values
    }
    if node.type == "conv3d" {
      guard schemaVersion == 2 else { throw GraphConfigurationError("Conv3d requires graph configuration schema 2.") }
      return try Conv3dConfiguration(values: node.config).values
    }
    if node.type == "conv2d" {
      guard schemaVersion == 2 else { throw GraphConfigurationError("Conv2d requires graph configuration schema 2.") }
      return try Conv2dConfiguration(values: node.config).values
    }
    if node.type == "conv1d" {
      guard schemaVersion == 2 else { throw GraphConfigurationError("Conv1d requires graph configuration schema 2.") }
      return try Conv1dConfiguration(values: node.config).values
    }
    if node.type == "lazy_linear" {
      guard schemaVersion == 2 else { throw GraphConfigurationError("LazyLinear requires graph configuration schema 2.") }
      return try LazyLinearConfiguration(values: node.config).values
    }
    if node.type == "bilinear" {
      guard schemaVersion == 2 else { throw GraphConfigurationError("Bilinear requires graph configuration schema 2.") }
      return try BilinearConfiguration(values: node.config).values
    }
    if ["bag_input","embedding_bag"].contains(node.type) {
      guard schemaVersion == 2 else {throw GraphConfigurationError("Bag components require graph configuration schema 2.")}
      if node.type == "bag_input" {return try BagSourceConfiguration(values:node.config).values}
      return try EmbeddingBagConfiguration(values:node.config).values
    }
    if ["index_input","embedding"].contains(node.type) {
      guard schemaVersion == 2 else {throw GraphConfigurationError("Integer lookup components require graph configuration schema 2.")}
      if node.type == "index_input" {return try IndexInputConfiguration(values:node.config).values}
      return try EmbeddingConfiguration(values:node.config).values
    }
    if ["reshape","transpose","flatten"].contains(node.type) {
      guard schemaVersion == 2 else {throw GraphConfigurationError("Tensor transforms require graph configuration schema 2.")}
      return try TensorTransformConfiguration(operation:node.type,values:node.config).values
    }
    if node.type == "softmax" {
      guard schemaVersion == 2 else {throw GraphConfigurationError("Softmax requires graph configuration schema 2.")}
      return try SoftmaxConfiguration(values:node.config).values
    }
    if node.type == "rms_norm" {
      guard schemaVersion == 2 else {throw GraphConfigurationError("RMS normalization requires graph schema2.")}
      return try RMSNormConfiguration(values:node.config).values
    }
    if node.type == "layer_norm" {
      guard schemaVersion == 2 else {throw GraphConfigurationError("Layer normalization requires graph configuration schema 2.")}
      return try LayerNormConfiguration(values:node.config).values
    }
    var config = node.config
    if schemaVersion == 1 && node.type == "linear" { config["bias"] = .bool(true) }
    return config
  }
  /// Explicit editing migration; Codable intentionally preserves historical versions and raw fields.
  public static func toCurrent(_ graph: GraphDocument, registry: ComponentRegistry) throws
    -> GraphDocument
  {
    guard [1, 2].contains(graph.schemaVersion) else {
      throw GraphConfigurationError("Unsupported graph schema version.")
    }
    var copy = graph
    for i in copy.nodes.indices {
      copy.nodes[i].config = try effectiveConfiguration(
        of: graph.nodes[i], schemaVersion: graph.schemaVersion, registry: registry)
    }
    copy.schemaVersion = 2
    return copy
  }
}

public enum GraphSemantics {
  /// Configuration-only validation supports incomplete editor drafts. Does not certify executable topology.
  public static func canonicalData(_ graph: GraphDocument, registry: ComponentRegistry) throws
    -> Data
  {
    guard [1, 2].contains(graph.schemaVersion) else {
      throw GraphConfigurationError("Unsupported graph schema version.")
    }
    let nodes = try graph.nodes.map { node -> [String: JSONValue] in
      var config = try GraphMigration.effectiveConfiguration(
        of: node, schemaVersion: graph.schemaVersion, registry: registry)
      if node.type == "mixture_experts" {config=MixtureExpertsConfiguration.scientificValues(config)}
      if node.type == "linear", config["bias"] == .bool(true) { config.removeValue(forKey: "bias") }
      // Match the worker's legacy semantic identity while retaining configured behavior changes.
      if node.type == "concat", config["dim"] == -1 { config.removeValue(forKey: "dim") }
      if node.type == "add" {
        if config["alpha"] == 1 { config.removeValue(forKey: "alpha") }
        if config["sample_broadcast"] == .bool(false) { config.removeValue(forKey: "sample_broadcast") }
      }
      return ["id": .string(node.id), "type": .string(node.type), "config": .object(config)]
    }.sorted { ($0["id"]?.stringValue ?? "") < ($1["id"]?.stringValue ?? "") }
    let edges = graph.edges.map {
      [
        "source": $0.source, "sourcePort": $0.sourcePort, "target": $0.target,
        "targetPort": $0.targetPort,
      ]
    }.sorted { a, b in
      let keys = ["source", "sourcePort", "target", "targetPort"]
      for key in keys { if a[key] != b[key] { return a[key]! < b[key]! } }
      return false
    }
    var fields: [String: JSONValue] = [
      "schemaVersion": 1, "nodes": .array(nodes.map(JSONValue.object)),
      "edges": .array(edges.map { .object($0.mapValues(JSONValue.string)) }),
    ]
    if let names=graph.namedInputs {fields["namedInputs"] = .object(names.mapValues(JSONValue.string))}
    if let plan = graph.compositionPlan {
      try plan.validate(graph: graph, registry: registry)
      fields["compositionPlan"] = try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(plan))
    }
    let value: JSONValue = .object(fields)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    return try encoder.encode(value)
  }
}
