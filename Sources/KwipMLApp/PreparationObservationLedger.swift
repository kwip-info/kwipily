import Foundation
import KwipMLCore

/// Captured in-process authority. This is not a portable graph or completed result.
struct PreparationObservationIdentity: Equatable {
    let requestID: String
    let graphSHA256: String
    let sourceBindingsSHA256: String
    let revision: String
}
struct PreparationColumnOrigin: Equatable {
    let nodeID: String
    let port: String
    let columnName: String
}
struct PreparationObservedColumn: Equatable {
    let schema: DataColumnSchema
    let origin: PreparationColumnOrigin
}
struct PreparationSchemaObservation: Equatable {
    let endpoint: DataFlowEndpoint
    let columns: [PreparationObservedColumn]
    let rowCount: Int64
    let sourceDigest: String
    let membershipDigest: String
    let dataVersion: String?
    let role: String
    let partitionDigest: String
    let fittedStateDigests: [JSONValue]
}
enum PreparationObservationStatus: String { case running, completed, failed, cancelled }

/// Bounded schema-only evidence from already supervised live preparation callbacks.
/// No cell values, machine paths or success artifacts are retained here.
struct PreparationObservationLedger {
    static let maximumEventBytes = 1_000_000
    static let maximumRetainedEventBytes = 32_000_000
    let identity: PreparationObservationIdentity
    private(set) var status: PreparationObservationStatus = .running
    private let graph: DataFlowDocument
    private let outputKinds: [String: [String: String]]
    private let expectedSourceDigests: [String: String]
    private var started = Set<String>()
    private var accepted = Set<String>()
    private var observations: [DataFlowEndpoint: PreparationSchemaObservation] = [:]
    private var retainedEventBytes = 0

    init(identity: PreparationObservationIdentity, graph: DataFlowDocument, outputKinds: [String: [String: String]], expectedSourceDigests: [String: String] = [:]) throws {
        try DataFlowCodec.validate(graph)
        guard !identity.requestID.isEmpty, identity.requestID.utf8.count <= 200,
              !identity.revision.isEmpty, identity.revision.utf8.count <= 200,
              WorkflowArchiveCodec.isHash(identity.graphSHA256), WorkflowArchiveCodec.isHash(identity.sourceBindingsSHA256),
              Set(outputKinds.keys) == Set(graph.nodes.map(\.id)),
              Set(expectedSourceDigests.keys) == Set(graph.nodes.filter { DataFlowSourceOperators.contains($0.operatorID) }.map(\.id)),
              expectedSourceDigests.values.allSatisfy(WorkflowArchiveCodec.isHash),
              outputKinds.values.allSatisfy({ ports in
                  (1...16).contains(ports.count) && ports.allSatisfy { port, kind in
                      !port.isEmpty && port.utf8.count <= 100 && port.unicodeScalars.allSatisfy { CharacterSet.alphanumerics.contains($0) || $0 == "_" || $0 == "-" }
                      && ["table", "dataset", "state"].contains(kind)
                  }
              }) else { throw DataFlowContractError("Observation authority or declared output bounds are invalid.") }
        self.identity = identity; self.graph = graph; self.outputKinds = outputKinds; self.expectedSourceDigests = expectedSourceDigests
    }

    @discardableResult mutating func nodeStarted(_ nodeID: String, identity: PreparationObservationIdentity) -> Bool {
        guard self.identity == identity, status == .running, outputKinds[nodeID] != nil,
              !accepted.contains(nodeID), !started.contains(nodeID) else { return false }
        started.insert(nodeID); return true
    }

    /// The caller must validate the current worker/phase callback before supplying authority.
    /// Every port is checked before any schema is published; rejected frames retain no state.
    @discardableResult mutating func consume(_ event: [String: JSONValue], identity: PreparationObservationIdentity) -> Bool {
        guard self.identity == identity, status == .running,
              event["event"] == .string("dataflowPreview"), case .string(let nodeID) = event["nodeID"],
              started.contains(nodeID), !accepted.contains(nodeID), let kinds = outputKinds[nodeID],
              let node = graph.nodes.first(where: { $0.id == nodeID }),
              case .object(let outputs) = event["outputs"], Set(outputs.keys) == Set(kinds.keys),
              case .object(let execution) = event["execution"], execution["nodeID"] == .string(nodeID),
              (execution["status"] == .string("executed") || execution["status"] == .string("reused")),
              let bytes = try? JSONEncoder().encode(event), bytes.count <= Self.maximumEventBytes,
              bytes.count <= Self.maximumRetainedEventBytes - retainedEventBytes else { return false }
        var staged: [DataFlowEndpoint: PreparationSchemaObservation] = [:]
        do {
            for (port, value) in outputs {
                guard let kind = kinds[port] else { return false }
                try WorkflowPreparationPreviewValidation.validate(value, kind: kind, operatorID: node.operatorID)
                func matches(_ source:String)->Bool {
                    if node.operatorID==TextEncodingConfiguration.operatorID,port=="vocabulary" {
                        guard case .object(let state)=outputs["state"],case .object(let summary)=state["summary"] else {return false}
                        return summary["contentDigest"] == .string(source)
                    }
                    return sourceMatches(source,node:node)
                }
                guard case .object(let output) = value, case .object(let qualifiedLineage) = output["lineage"],
                      case .string(let qualifiedSource) = qualifiedLineage["sourceDigest"], matches(qualifiedSource) else { return false }
                guard kind != "state" else { continue }
                guard case .object(let preview) = output["preview"],
                      case .array(let columns) = preview["columns"], case .integer(let rows) = preview["rowCount"],
                      case .object(let lineage) = output["lineage"], case .string(let source) = lineage["sourceDigest"],
                      case .string(let membership) = lineage["membershipDigest"],
                      case .string(let role) = lineage["role"], case .string(let partition) = lineage["partitionDigest"],
                      case .array(let fitted) = lineage["fittedStateDigests"] else { return false }
                guard matches(source) else { return false }
                let version:String?
                if case .string(let pin)=lineage["dataVersion"] {version=pin}
                else if output["schemaVersion"] == .integer(2),lineage["dataVersion"] == .null {version=nil}
                else {return false}
                if node.operatorID == "kwip.data.reshapeTable" || (node.operatorID == "kwip.data.splitText" && node.config["mode"] == .string("delimited") && node.config["outputMode"] == .string("rows")) {
                    let inputs = graph.edges.filter { $0.to.node == node.id && $0.to.port == "table" }
                    guard inputs.count == 1, let upstream = observations[inputs[0].from], upstream.role == "unassigned", upstream.partitionDigest.isEmpty, upstream.fittedStateDigests.isEmpty, role == "unassigned", partition.isEmpty, fitted.isEmpty else { return false }
                }
                let endpoint = DataFlowEndpoint(node: nodeID, port: port)
                let observed: [PreparationObservedColumn] = try columns.map { value in
                    guard case .object(let fields) = value, case .string(let name) = fields["name"],
                          case .string(let dtype) = fields["dtype"], case .bool(let nullable) = fields["nullable"] else {
                        throw DataFlowContractError("Observed column shape is invalid.")
                    }
                    return .init(schema: .init(name: name, dtype: dtype, nullable: nullable, temporal: { if case .object(let metadata) = fields["temporal"] { return metadata }; return nil }()), origin: columnOrigin(name, node: node, endpoint: endpoint))
                }
                staged[endpoint] = .init(endpoint: endpoint, columns: observed, rowCount: rows, sourceDigest: source, membershipDigest: membership, dataVersion: version, role: role, partitionDigest: partition, fittedStateDigests: fitted)
            }
        } catch { return false }
        observations.merge(staged) { _, new in new }
        retainedEventBytes += bytes.count; accepted.insert(nodeID); return true
    }

    private func sourceMatches(_ source: String, node: DataFlowNode) -> Bool {
        if DataFlowSourceOperators.contains(node.operatorID) { return expectedSourceDigests[node.id] == source }
        if node.operatorID == GameRuleConfiguration.operatorID,let definition=node.config["definition"] {
            return source == pythonDigest(definition)
        }
        if node.operatorID == "kwip.data.union" {
            guard let config=try? UnionConfiguration(node.config) else {return false}
            var parents:[JSONValue]=[]
            for port in config.inputs {
                let edges=graph.edges.filter {$0.to.node == node.id && $0.to.port == port}
                guard edges.count == 1,let digest=observations[edges[0].from]?.sourceDigest else {return false}
                parents.append(.array([.string(port),.string(digest)]))
            }
            return source == pythonDigest(.array([.string("kwip.union.sources.v1"),.array(parents)]))
        }
        if node.operatorID == "kwip.data.join" {
            let left = graph.edges.filter { $0.to.node == node.id && $0.to.port == "left" }
            let right = graph.edges.filter { $0.to.node == node.id && $0.to.port == "right" }
            guard left.count == 1, right.count == 1,
                  let leftSource = observations[left[0].from]?.sourceDigest,
                  let rightSource = observations[right[0].from]?.sourceDigest else { return false }
            return source == pythonDigest(.array([.string("kwip.join.sources.v1"), .string(leftSource), .string(rightSource)]))
        }
        if node.operatorID == "kwip.data.reshapeTable" || (node.operatorID == "kwip.data.splitText" && node.config["mode"] == .string("delimited") && node.config["outputMode"] == .string("rows")) {
            let input = graph.edges.filter { $0.to.node == node.id && $0.to.port == "table" }
            guard input.count == 1, let original = observations[input[0].from]?.sourceDigest else { return false }
            let tag = node.operatorID == "kwip.data.reshapeTable" ? "kwip.reshape.sources.v1" : "kwip.splitText.rows.v1"
            return source == pythonDigest(.array([.string(tag), .string(original), .object(node.config)]))
        }
        if node.operatorID == "kwip.data.aggregate" {
            let input = graph.edges.filter { $0.to.node == node.id && $0.to.port == "table" }
            guard input.count == 1, let original = observations[input[0].from]?.sourceDigest,
                  let groups = node.config["groupBy"], let nulls = node.config["nullKeys"] else { return false }
            return source == pythonDigest(.array([.string("kwip.aggregate.sources.v1"), .string(original), groups, nulls]))
        }
        let inputs = graph.edges.filter { $0.to.node == node.id && ["table", "training"].contains($0.to.port) }
        guard inputs.count == 1, let input = inputs.first, let upstream = observations[input.from] else { return false }
        return upstream.sourceDigest == source
    }

    /// Match Python json.dumps(sort_keys=True,separators=(",",":")) including
    /// explicit floating-point literals in Reshape's typed fixed fill.
    private func pythonDigest(_ value: JSONValue) -> String? {
        func literal(_ value: JSONValue) throws -> String {
            switch value {
            case .null: return "null"
            case .bool(let flag): return flag ? "true" : "false"
            case .integer(let integer): return String(integer)
            case .number(let number):
                guard number.isFinite else { throw DataFlowContractError("Nonfinite source identity configuration.") }
                return pythonFloat(number)
            case .string:
                let encoder = JSONEncoder(); encoder.outputFormatting = [.withoutEscapingSlashes]
                let text = String(decoding: try encoder.encode(value), as: UTF8.self)
                var ascii = ""
                for scalar in text.unicodeScalars {
                    let code = scalar.value
                    if code < 127 { ascii.unicodeScalars.append(scalar) }
                    else if code <= 0xffff { ascii += String(format: "\\u%04x", code) }
                    else {
                        let rest = code - 0x10000
                        ascii += String(format: "\\u%04x\\u%04x", 0xd800 + (rest >> 10), 0xdc00 + (rest & 0x3ff))
                    }
                }
                return ascii
            case .array(let values): return "[" + (try values.map(literal)).joined(separator: ",") + "]"
            case .object(let values):
                let keys = values.keys.sorted { $0.utf8.lexicographicallyPrecedes($1.utf8) }
                return "{" + (try keys.map { try literal(.string($0)) + ":" + literal(values[$0]!) }).joined(separator: ",") + "}"
            }
        }
        guard let text = try? literal(value) else { return nil }
        return WorkflowArchiveCodec.hash(Data(text.utf8))
    }
    private func pythonFloat(_ value: Double) -> String {
        let description = String(value).lowercased()
        if value == 0 { return value.sign == .minus ? "-0.0" : "0.0" }
        let negative = description.hasPrefix("-")
        let unsigned = negative ? String(description.dropFirst()) : description
        let pieces = unsigned.split(separator: "e", omittingEmptySubsequences: false)
        let mantissa = String(pieces[0]), explicitExponent = pieces.count == 2 ? Int(pieces[1]) ?? 0 : 0
        let decimal = mantissa.firstIndex(of: ".")
        let before = decimal.map { mantissa.distance(from: mantissa.startIndex, to: $0) } ?? mantissa.count
        var digits = mantissa.filter { $0 != "." }
        var exponent = explicitExponent + before - 1
        while digits.first == "0" { digits.removeFirst(); exponent -= 1 }
        while digits.count > 1 && digits.last == "0" { digits.removeLast() }
        let sign = negative ? "-" : ""
        if exponent < -4 || exponent >= 16 {
            let tail = digits.dropFirst()
            let significand = String(digits.prefix(1)) + (tail.isEmpty ? "" : "." + String(tail))
            return sign + significand + "e" + (exponent < 0 ? "-" : "+") + String(format: "%02d", abs(exponent))
        }
        let point = exponent + 1
        if point <= 0 { return sign + "0." + String(repeating: "0", count: -point) + digits }
        if point >= digits.count { return sign + digits + String(repeating: "0", count: point - digits.count) + ".0" }
        let split = digits.index(digits.startIndex, offsetBy: point)
        return sign + String(digits[..<split]) + "." + String(digits[split...])
    }

    private func columnOrigin(_ name: String, node: DataFlowNode, endpoint: DataFlowEndpoint) -> PreparationColumnOrigin {
        let fallback = PreparationColumnOrigin(nodeID: endpoint.node, port: endpoint.port, columnName: name)
        let inputs = graph.edges.filter { $0.to.node == node.id && ["table", "training"].contains($0.to.port) }
        guard inputs.count == 1, let input = inputs.first, let previous = observations[input.from] else { return fallback }
        var upstreamName = name
        if node.operatorID == "kwip.data.selectRename" {
            guard case .array(let selected) = node.config["columns"], case .object(let rename) = node.config["rename"] else { return fallback }
            let names = selected.compactMap { if case .string(let value) = $0 { return value }; return nil }
            let matches = names.filter { source in (rename[source] ?? .string(source)) == .string(name) }
            guard matches.count == 1, let original = matches.first else { return fallback }
            upstreamName = original
        }
        if node.operatorID == "kwip.data.columns", case .array(let rows) = node.config["rows"] {
            let matches = rows.compactMap { value -> String? in
                guard case .object(let row) = value, row["included"] == .bool(true), row["outputName"] == .string(name), case .string(let source) = row["source"] else { return nil }
                return source
            }
            if matches.count == 1 { upstreamName = matches[0] }
            else if !matches.isEmpty { return fallback }
        }
        // New/overwritten derived columns begin at this operator; untouched columns
        // retain the upstream origin. Never infer origin by column position.
        let preserving = ["kwip.data.sort", "kwip.data.unique", "kwip.data.splitText", "kwip.data.filterRows", "kwip.data.columns", "kwip.data.selectRename", "kwip.data.cast", "kwip.data.filter", "kwip.data.filterExpression", "kwip.data.regexMatch", "kwip.data.split", "kwip.data.groupSplit", "kwip.data.temporalSplit", "kwip.data.sampleRows", "kwip.data.outputDelimited", "kwip.data.view"]
        if node.operatorID == "kwip.data.splitText" {
            guard case .array(let fields) = node.config["fields"] else { return fallback }
            let rowMode = node.config["mode"] == .string("delimited") && node.config["outputMode"] == .string("rows")
            let active = rowMode ? Array(fields.prefix(1)) : fields
            guard !active.contains(where: { if case .object(let field) = $0 { return field["name"] == .string(name) }; return true }), !(rowMode && node.config["ordinal"] == .bool(true) && node.config["ordinalName"] == .string(name)) else { return fallback }
        }
        if ["kwip.data.calculate", "kwip.data.regexReplace"].contains(node.operatorID) {
            guard node.config["outputColumn"] != .string(name) else { return fallback }
        } else if node.operatorID == "kwip.data.regexExtract" {
            guard case .array(let groups) = node.config["groups"], !groups.contains(where: {
                if case .object(let group) = $0 { return group["output"] == .string(name) }; return true
            }) else { return fallback }
        } else if !preserving.contains(node.operatorID) { return fallback }
        return previous.columns.first { $0.schema.name == upstreamName }?.origin ?? fallback
    }

    mutating func finish(_ terminal: PreparationObservationStatus) {
        guard status == .running, [.failed, .cancelled].contains(terminal) else { return }
        status = terminal
    }
    /// Successful archive verification is an external prerequisite; previews alone never complete.
    @discardableResult mutating func completeVerified(identity: PreparationObservationIdentity, allOutputsVerified: Bool) -> Bool {
        guard self.identity == identity, status == .running, allOutputsVerified,
              accepted == Set(outputKinds.keys) else { return false }
        status = .completed; return true
    }
    func observation(at endpoint: DataFlowEndpoint, current: PreparationObservationIdentity) -> PreparationSchemaObservation? {
        guard identity == current else { return nil }
        return observations[endpoint]
    }
    func disclosure(current: PreparationObservationIdentity) -> String {
        guard identity == current else { return "Stale observed columns · source or flow changed; run again to refresh." }
        switch status {
        case .running: return "Observed columns · current run in progress; reached outputs only."
        case .completed: return "Observed columns · verified completed preparation; sources not revalidated here."
        case .failed: return "Partial observed columns · run failed; reached outputs only."
        case .cancelled: return "Partial observed columns · run cancelled; reached outputs only."
        }
    }
}
