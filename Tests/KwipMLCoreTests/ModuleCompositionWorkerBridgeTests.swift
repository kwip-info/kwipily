import XCTest
import Darwin
@testable import KwipMLCore

/// Consume actual native expansion bytes in the actual Python model, rather than
/// comparing independently hand-authored plans that might agree on the same mistake.
final class ModuleCompositionWorkerBridgeTests: XCTestCase {
    func testEveryAuthoredContainerExecutesWithCrossLanguageIdentityAndOwners() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python = root.appendingPathComponent(".venv/bin/python")
        XCTAssertTrue(FileManager.default.isExecutableFile(atPath: python.path))
        let script = """
import copy,hashlib,json,sys,torch
from runner.engine import GraphModel
from runner.graph_semantics import semantic_graph
from runner.native_ownership import native_owners
from runner.composite_source import checked_source
payload=json.load(sys.stdin);graph=payload['graph'];roles=payload['roles']
assert semantic_graph(graph)==payload['semantic']
source=payload['source'];raw=json.dumps(source,ensure_ascii=False).encode('utf8')
proof=checked_source(raw,hashlib.sha256(raw).hexdigest(),graph,registry='Sources/KwipMLCore/Resources/components.json')
assert proof['nativeExpansionContractVersion']==2
# A fresh hash is not proof that an origin still came from the captured source.
forged=copy.deepcopy(source);forged['nodeOrigins'][roles['a']]['localNodeID']='forged'
raw_forged=json.dumps(forged,ensure_ascii=False).encode('utf8')
try:checked_source(raw_forged,hashlib.sha256(raw_forged).hexdigest(),graph,registry='Sources/KwipMLCore/Resources/components.json')
except ValueError:pass
else:raise AssertionError('Rehashed forged native origin accepted')
devices=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])
for device in devices:
 model=GraphModel(graph).to(device)
 owners=native_owners(model)
 assert len(owners)==4 and len({id(p) for _,p in owners})==4
 with torch.no_grad():
  for meta,p in owners:
   first=meta['node']==roles['a'];assert first or meta['node']==roles['b']
   assert (meta['name'].startswith('composition_root.') and '.operation.' in meta['name']) if 'compositionPlan' in graph else meta['name'].startswith('layers.')
   if meta['parameter']=='weight':p.copy_(torch.eye(2,device=device)*(2 if first else 3))
   else:
    assert meta['parameter']=='bias';p.fill_(1 if first else -2)
 x=torch.tensor([[1.,-2.],[0.,3.]],device=device,requires_grad=True)
 y=model(x)
 assert torch.equal(y.detach().cpu(),torch.tensor([[7.,-11.],[1.,19.]]))
 y.sum().backward();assert torch.equal(x.grad.cpu(),torch.full((2,2),6.))
 for meta,p in owners:
  first=meta['node']==roles['a']
  expected=torch.full((2,2),3. if first else 4.) if meta['parameter']=='weight' else torch.full((2,),6. if first else 2.)
  assert torch.equal(p.grad.cpu(),expected),(meta,p.grad,expected)
print(json.dumps({'devices':devices,'owners':4,'verified':True}))
"""
        let variants = CompositionPolicy.Kind.allCases.map { ($0, true) } + [(.graph, false)]
        for (kind, reachable) in variants {
            let ref = CompositeDefinitionReference("two-steps")
            let children = kind == .sequential ? ["a", "b"] : ["b", "a"]
            var definition = CompositeDefinition(reference: ref, title: "Two steps", body: .init(nodes: [
                .init(id: "a", type: "linear", config: ["in_features": 2, "out_features": 2]),
                .init(id: "b", type: "linear", config: ["in_features": 2, "out_features": 2])
            ], edges: [.init(id: "ab", source: "a", target: "b")]), inputs: [.init("values", targets: [.init("a", "in")])], outputs: [.init("result", source: .init("b", "out"))], composition: .init(kind: kind, children: children.map { .init(localID: $0, key: kind == .namedMap ? "step_" + $0 : nil) }))
            var definitions = [definition]
            if !reachable {
                definition.composition = nil; definitions = [definition]
                var unused = definition; unused.reference = .init("unused")
                unused.composition = .init(kind: .graph, children: [.init(localID: "a"), .init(localID: "b")])
                definitions.append(unused)
            }
            let source = CompositeProjectDocument(schemaVersion: 3, root: .init(nodes: [
                .init(id: "input", type: "input", config: ["features": 2]), .init(id: "output", type: "output")
            ], instances: [.init(id: "block", definition: ref)], edges: [
                .init(id: "enter", source: "input", target: "block", targetPort: "values"),
                .init(id: "leave", source: "block", sourcePort: "result", target: "output")
            ]), definitions: definitions)
            let registry = try ComponentRegistry.builtIn(), expansion = try CompositeExpander.expand(source, registry: registry)
            let encoder = JSONEncoder()
            let graph = try JSONDecoder().decode(JSONValue.self, from: encoder.encode(expansion.graph))
            let snapshot = try JSONDecoder().decode(JSONValue.self, from: encoder.encode(CompositeSourceSnapshot(project: source, expansion: expansion)))
            let semantic = try JSONDecoder().decode(JSONValue.self, from: GraphSemantics.canonicalData(expansion.graph, registry: registry))
            let roles = Dictionary(uniqueKeysWithValues: expansion.nodeOrigins.filter { ["a", "b"].contains($0.value.localNodeID) }.map { ($0.value.localNodeID, JSONValue.string($0.key)) })
            let input = try encoder.encode(JSONValue.object(["graph": graph, "semantic": semantic, "source": snapshot, "roles": .object(roles)]))
            let process = Process(), stdin = Pipe(), stdout = Pipe()
            process.executableURL = python; process.currentDirectoryURL = root; process.arguments = ["-B", "-c", script]
            process.standardInput = stdin; process.standardOutput = stdout; process.standardError = stdout
            try process.run(); try stdin.fileHandleForWriting.write(contentsOf: input); try stdin.fileHandleForWriting.close()
            let deadline = Date().addingTimeInterval(25)
            while process.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.025) }
            if process.isRunning { _ = Darwin.kill(process.processIdentifier, SIGKILL) }
            process.waitUntilExit()
            let log = String(decoding: stdout.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            XCTAssertEqual(process.terminationStatus, 0, "\(kind): \(log)")
            XCTAssertTrue(log.contains("\"verified\": true"), "\(kind): \(log)")
        }
    }
}
