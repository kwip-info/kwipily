import Foundation
import XCTest
@testable import KwipMLCore

final class ModuleCompositionPolicyTests: XCTestCase {
    private let ports: [String: CompositionPolicy.ChildPorts] = [
        "a": .init(inputs: ["in"], outputs: ["out"]),
        "b": .init(inputs: ["in"], outputs: ["out"])
    ]
    private var chain: [GraphEdge] { [.init(id: "ab", source: "a", target: "b")] }
    private var inputs: [CompositeInput] { [.init("input", targets: [.init("a", "in")])] }
    private var outputs: [CompositeOutput] { [.init("result", source: .init("b", "out"))] }
    private func policy(_ kind: CompositionPolicy.Kind = .sequential, _ children: [CompositionPolicy.Child] = [.init(localID: "a"), .init(localID: "b")]) -> CompositionPolicy {
        .init(kind: kind, children: children)
    }
    private func validate(_ p: CompositionPolicy, edges: [GraphEdge]? = nil, inputs: [CompositeInput]? = nil, outputs: [CompositeOutput]? = nil, ports: [String: CompositionPolicy.ChildPorts]? = nil) throws {
        try p.validate(childPorts: ports ?? self.ports, edges: edges ?? chain, inputs: inputs ?? self.inputs, outputs: outputs ?? self.outputs)
    }
    func testLiteralSequentialOrderRequiresExactUnaryChain() throws {
        try validate(policy())
        XCTAssertThrowsError(try validate(policy(.sequential, [.init(localID: "b"), .init(localID: "a")])))
        XCTAssertThrowsError(try validate(policy(), edges: []))
        XCTAssertThrowsError(try validate(policy(), edges: chain + [.init(id: "ba", source: "b", target: "a")]))
        XCTAssertThrowsError(try validate(policy(), inputs: [.init("input", targets: [.init("a", "in"), .init("b", "in")])]))
        XCTAssertThrowsError(try validate(policy(), outputs: [.init("result", source: .init("a", "out"))]))
        XCTAssertThrowsError(try validate(policy(), ports: ["a": .init(inputs: ["in", "state"], outputs: ["out"]), "b": ports["b"]!]))
    }
    func testExactDirectChildrenAndBoundsDoNotSilentlyRepair() throws {
        let invalidChildren: [[CompositionPolicy.Child]] = [[], [.init(localID: "a")], [.init(localID: "a"), .init(localID: "a")], [.init(localID: "a"), .init(localID: "foreign")]]
        for children in invalidChildren {
            XCTAssertThrowsError(try validate(policy(.graph, children)))
        }
        XCTAssertThrowsError(try validate(.init(schemaVersion: 2, kind: .graph, children: policy().children)))
        let many = (0..<129).map { CompositionPolicy.Child(localID: "n\($0)") }
        let manyPorts = Dictionary(uniqueKeysWithValues: many.map { ($0.localID, CompositionPolicy.ChildPorts(inputs: ["in"], outputs: ["out"])) })
        XCTAssertThrowsError(try validate(policy(.orderedList, many), edges: [], inputs: [], outputs: [], ports: manyPorts))
    }
    func testNamedMapKeysAreExplicitUniqueAndNativeSafe() throws {
        try validate(policy(.namedMap, [.init(localID: "b", key: "second"), .init(localID: "a", key: "first")]))
        let invalidKeys: [String?] = [nil, "", "first.second", "1first", "has space", "training", "forward", "_modules", String(repeating: "x", count: 65), "é"]
        for key in invalidKeys {
            XCTAssertThrowsError(try validate(policy(.namedMap, [.init(localID: "a", key: key), .init(localID: "b", key: "other")])))
        }
        XCTAssertThrowsError(try validate(policy(.namedMap, [.init(localID: "a", key: "same"), .init(localID: "b", key: "same")])))
        for kind in [CompositionPolicy.Kind.graph, .sequential, .orderedList] {
            XCTAssertThrowsError(try validate(policy(kind, [.init(localID: "a", key: "first"), .init(localID: "b")])))
        }
    }
    func testGraphAndOrderedContainersKeepExplicitDagButRejectForeignPorts() throws {
        for kind in [CompositionPolicy.Kind.graph, .orderedList, .namedMap] {
            let children: [CompositionPolicy.Child] = kind == .namedMap ? [.init(localID: "b", key: "b"), .init(localID: "a", key: "a")] : [.init(localID: "b"), .init(localID: "a")]
            try validate(policy(kind, children)) // Registration order need not be topological.
            XCTAssertThrowsError(try validate(policy(kind, children), edges: [.init(id: "bad", source: "a", sourcePort: "hidden", target: "b")]))
            XCTAssertThrowsError(try validate(policy(kind, children), inputs: [.init("input", targets: [.init("foreign", "in")])]))
            XCTAssertThrowsError(try validate(policy(kind, children), outputs: [.init("result", source: .init("b", "missing"))]))
        }
    }
    func testStrictWireUnknownNullAndWrongTypesRefuse() throws {
        let good = #"{"schemaVersion":1,"kind":"graph","children":[{"localID":"a"},{"localID":"b"}]}"#
        let decoded = try JSONDecoder().decode(CompositionPolicy.self, from: Data(good.utf8))
        XCTAssertEqual(decoded, policy(.graph))
        for bad in [
            #"{"schemaVersion":1,"kind":"graph","children":[{"localID":"a"}],"script":"x"}"#,
            #"{"schemaVersion":1,"kind":"graph","children":[{"localID":"a","key":null}]}"#,
            #"{"schemaVersion":1,"kind":"graph","children":[{"localID":"a","extra":1}]}"#,
            #"{"schemaVersion":true,"kind":"graph","children":[]}"#,
            #"{"schemaVersion":null,"kind":"graph","children":[]}"#,
            #"{"schemaVersion":2,"kind":"graph","children":[{"localID":"a"}]}"#,
            #"{"schemaVersion":1,"kind":"code","children":[{"localID":"a"}]}"#
        ] { XCTAssertThrowsError(try JSONDecoder().decode(CompositionPolicy.self, from: Data(bad.utf8)), bad) }
    }
    func testReviewedNativeReservedKeysAndSafeUnderscoreParity() throws {
        // Literal installed Torch 2.14.1 Module/ModuleDict snapshot, independently
        // captured by the runtime owner before this Core validation implementation.
        let reviewed: Set<String> = [
            "T_destination",
            "__annotations__",
            "__call__",
            "__class__",
            "__contains__",
            "__delattr__",
            "__delitem__",
            "__dict__",
            "__dir__",
            "__doc__",
            "__eq__",
            "__firstlineno__",
            "__format__",
            "__ge__",
            "__getattr__",
            "__getattribute__",
            "__getitem__",
            "__getstate__",
            "__gt__",
            "__hash__",
            "__init__",
            "__init_subclass__",
            "__iter__",
            "__le__",
            "__len__",
            "__lt__",
            "__module__",
            "__ne__",
            "__new__",
            "__reduce__",
            "__reduce_ex__",
            "__repr__",
            "__setattr__",
            "__setitem__",
            "__setstate__",
            "__sizeof__",
            "__static_attributes__",
            "__str__",
            "__subclasshook__",
            "__weakref__",
            "_apply",
            "_backward_hooks",
            "_backward_pre_hooks",
            "_buffers",
            "_call_impl",
            "_compiled_call_impl",
            "_forward_hooks",
            "_forward_hooks_always_called",
            "_forward_hooks_with_kwargs",
            "_forward_pre_hooks",
            "_forward_pre_hooks_with_kwargs",
            "_get_backward_hooks",
            "_get_backward_pre_hooks",
            "_get_name",
            "_is_full_backward_hook",
            "_load_from_state_dict",
            "_load_state_dict_post_hooks",
            "_load_state_dict_pre_hooks",
            "_maybe_warn_non_full_backward_hook",
            "_modules",
            "_named_members",
            "_non_persistent_buffers_set",
            "_parameters",
            "_register_load_state_dict_pre_hook",
            "_register_state_dict_hook",
            "_replicate_for_data_parallel",
            "_save_to_state_dict",
            "_slow_forward",
            "_state_dict_hooks",
            "_state_dict_pre_hooks",
            "_version",
            "_wrapped_call_impl",
            "add_module",
            "apply",
            "bfloat16",
            "buffers",
            "call_super_init",
            "children",
            "clear",
            "compile",
            "cpu",
            "cuda",
            "double",
            "dump_patches",
            "eval",
            "extra_repr",
            "float",
            "forward",
            "get_buffer",
            "get_extra_state",
            "get_parameter",
            "get_submodule",
            "half",
            "ipu",
            "items",
            "keys",
            "load_state_dict",
            "modules",
            "mtia",
            "named_buffers",
            "named_children",
            "named_modules",
            "named_parameters",
            "parameters",
            "pop",
            "register_backward_hook",
            "register_buffer",
            "register_forward_hook",
            "register_forward_pre_hook",
            "register_full_backward_hook",
            "register_full_backward_pre_hook",
            "register_load_state_dict_post_hook",
            "register_load_state_dict_pre_hook",
            "register_module",
            "register_parameter",
            "register_state_dict_post_hook",
            "register_state_dict_pre_hook",
            "requires_grad_",
            "set_extra_state",
            "set_submodule",
            "share_memory",
            "state_dict",
            "to",
            "to_empty",
            "train",
            "training",
            "type",
            "update",
            "values",
            "xpu",
            "zero_grad"
        ]
        XCTAssertEqual(reviewed.count, 131)
        XCTAssertEqual(CompositionPolicy.nativeReservedKeys, reviewed)
        for key in reviewed {
            XCTAssertThrowsError(try validate(policy(.namedMap, [.init(localID: "a", key: key), .init(localID: "b", key: "other")])), key)
        }
        try validate(policy(.namedMap, [.init(localID: "a", key: "_stage"), .init(localID: "b", key: "other")]))
    }
    func testLegacyDefinitionOmissionAndExplicitPolicyRoundTrip() throws {
        let old = CompositeDefinition(reference: .init("block"), title: "Block", body: .init(), inputs: [], outputs: [])
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let bytes = try encoder.encode(old)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
        XCTAssertEqual(Set(object.keys), ["reference", "title", "body", "inputs", "outputs", "parameters"])
        XCTAssertEqual(try encoder.encode(JSONDecoder().decode(CompositeDefinition.self, from: bytes)), bytes)
        var selected = old; selected.composition = policy(.graph)
        XCTAssertEqual(try JSONDecoder().decode(CompositeDefinition.self, from: encoder.encode(selected)), selected)
        XCTAssertNotEqual(try encoder.encode(selected), bytes)
        var explicitNull = object; explicitNull["composition"] = NSNull()
        XCTAssertThrowsError(try JSONDecoder().decode(CompositeDefinition.self, from: JSONSerialization.data(withJSONObject: explicitNull)))

    }
}
