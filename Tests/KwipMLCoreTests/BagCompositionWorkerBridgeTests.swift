import XCTest
import Darwin
@testable import KwipMLCore

/// Genuine Swift expansion and registry bytes cross the worker trust boundary.
/// Independent literal arithmetic grades the actual registered native table.
final class BagCompositionWorkerBridgeTests: XCTestCase {
    func testAllThirteenEligibleNativeBagSourcesCrossIndependentWorkerBoundary() throws {
        let fixtures = BagCompositionSourceTests(), registry = try fixtures.registry()
        let variants = [CompositionPolicy.Kind.graph, .orderedList, .namedMap].flatMap { kind in
            [false, true].flatMap { packed in [false, true].map { weighted in (kind, packed, weighted) } }
        } + [(.sequential, false, false)]
        for (kind, packed, weighted) in variants {
            let project = fixtures.project(kind: kind, packed: packed, weighted: weighted)
            let expansion = try CompositeExpander.expand(project, registry: registry)
            let snapshot = CompositeSourceSnapshot(project: project, expansion: expansion)
            let encoder = JSONEncoder()
            func json<T: Encodable>(_ value: T) throws -> JSONValue {
                try JSONDecoder().decode(JSONValue.self, from: encoder.encode(value))
            }
            let table = try XCTUnwrap(expansion.nodeOrigins.first { $0.value.localNodeID == "table" }?.key)
            let payload = try encoder.encode(JSONValue.object([
                "source": try json(snapshot), "registry": try json(registry),
                "table": .string(table), "kind": .string(kind.rawValue),
                "packed": .bool(packed), "weighted": .bool(weighted)
            ]))
            let log = try worker(payload)
            XCTAssertTrue(log.contains("\"verified\": true"), "\(kind)/\(packed)/\(weighted): \(log)")
        }
        XCTAssertEqual(variants.count, 13)
    }

    private func worker(_ input: Data) throws -> String {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python = root.appendingPathComponent(".venv/bin/python")
        XCTAssertTrue(FileManager.default.isExecutableFile(atPath: python.path))
        let process = Process(), stdin = Pipe(), stdout = Pipe()
        process.executableURL = python; process.currentDirectoryURL = root
        process.arguments = ["-B", "-c", Self.script]
        process.standardInput = stdin; process.standardOutput = stdout; process.standardError = stdout
        try process.run(); try stdin.fileHandleForWriting.write(contentsOf: input); try stdin.fileHandleForWriting.close()
        let deadline = Date().addingTimeInterval(30)
        while process.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.025) }
        if process.isRunning { _ = Darwin.kill(process.processIdentifier, SIGKILL) }
        process.waitUntilExit()
        let log = String(decoding: stdout.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        XCTAssertEqual(process.terminationStatus, 0, log)
        return log
    }

    private static let script = #"""
import copy,hashlib,importlib.util,json,sys,tempfile
from pathlib import Path
import torch
from runner.composite_source import checked_source
from runner.engine import GraphModel,export_source
from runner.embedding_bag_packet import EmbeddingBagPacket,collate_bags
from runner.native_ownership import native_owners,composition_identity
p=json.load(sys.stdin);source=p['source'];graph=source['expandedGraph'];table=p['table']
with tempfile.TemporaryDirectory(prefix='kwip-bag-core-worker-') as directory:
 root=Path(directory);registry=root/'registry.json';registry.write_text(json.dumps(p['registry']))
 def verify(value,actual=graph):
  raw=json.dumps(value,ensure_ascii=False,separators=(',',':')).encode()
  return checked_source(raw,hashlib.sha256(raw).hexdigest(),actual,registry=registry)
 assert verify(source)['nativeExpansionContractVersion']==2
 for mutation in ['origin','rootTargets','sourceRole','nestedAlias']:
  forged=copy.deepcopy(source)
  if mutation=='origin':forged['nodeOrigins'][table]['localNodeID']='forged'
  elif mutation=='rootTargets':forged['expandedGraph']['compositionPlan']['containers'][0]['inputs'][0]['targets'].pop()
  elif mutation=='sourceRole':forged['expandedGraph']['edges'][0]['sourcePort']='disabled'
  else:
   block=forged['expandedGraph']['compositionPlan']['containers'][1]
   block['inputs'][0]['id']='forged-name'
  try:verify(forged,forged['expandedGraph'])
  except ValueError:pass
  else:raise AssertionError('Rehashed forged capture admitted: '+mutation)
 devices=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])
 literal=[[9.,9.],[1.,2.],[3.,-1.],[-2.,4.],[3.,4.]]
 ids=[1,2,1,3] if p['packed'] else [[1,2,1],[0,0,0],[3,3,3]]
 weights=[.5,2.,-1.,1.] if p['packed'] else [[.5,2.,-1.],[0.,0.,0.],[1.,0.,0.]]
 bags=[[1,2,1],[],[3]] if p['packed'] else ids
 factors=[[.5,2.,-1.],[],[1.]] if p['packed'] else weights
 upstream=[[2.,-1.],[7.,11.],[-3.,5.]]
 expected=[];table_grad=[[0.,0.] for _ in literal];weight_grad=[]
 for r,bag in enumerate(bags):
  row=[0.,0.];dw=[];denominator=1 if p['weighted'] else len(bag) or 1
  for j,index in enumerate(bag):
   scale=factors[r][j] if p['weighted'] else 1/denominator
   for d in range(2):
    row[d]+=literal[index][d]*scale
    table_grad[index][d]+=upstream[r][d]*scale
   dw.append(sum(literal[index][d]*upstream[r][d] for d in range(2)))
  expected.append(row);weight_grad.append(dw)
 expected_weights=[v for bag in weight_grad for v in bag] if p['packed'] else weight_grad
 for device in devices:
  model=GraphModel(graph,registry=registry,bundled_dataset=False).to(device)
  owners=native_owners(model);assert len(owners)==1 and owners[0][0]['node']==table
  metadata,parameter=owners[0];assert metadata['parameter']=='weight' and metadata['name'].count('.operation.')==1
  assert len(composition_identity(model)['stateOwnership'])==1
  with torch.no_grad():parameter.copy_(torch.tensor(literal,device=device))
  actual_weights=torch.tensor(weights,device=device,requires_grad=True) if p['weighted'] else None
  item=EmbeddingBagPacket('packed' if p['packed'] else 'fixed',torch.tensor(ids,device=device),torch.tensor([0,3,3,4],device=device) if p['packed'] else None,actual_weights,p['packed'])
  block=model.get_submodule('composition_root.blocks.0')
  assert isinstance(block,{'graph':torch.nn.Module,'sequential':torch.nn.Sequential,'orderedList':torch.nn.ModuleList,'namedMap':torch.nn.ModuleDict}[p['kind']])
  calls=[];hook=block.register_forward_hook(lambda *a:calls.append(1))
  output,stages=model(item,inspect=True);hook.remove();assert calls==[1]
  torch.testing.assert_close(output,torch.tensor(expected,device=device),atol=2e-6,rtol=2e-6)
  for role,tensor in stages['bags'].items():assert tensor is getattr(item,role)
  (output*torch.tensor(upstream,device=device)).sum().backward()
  torch.testing.assert_close(parameter.grad,torch.tensor(table_grad,device=device),atol=2e-6,rtol=2e-6)
  if p['weighted']:torch.testing.assert_close(actual_weights.grad,torch.tensor(expected_weights,device=device))
  path=root/'model.py';path.write_text(export_source(model));torch.save(model.state_dict(),root/'weights.pt')
  spec=importlib.util.spec_from_file_location('core_bag_portable',path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  portable=module.Model().to(device);portable.load_state_dict(torch.load(root/'weights.pt',weights_only=True,map_location=device),strict=True)
  assert list(portable.state_dict())==list(model.state_dict())
  torch.testing.assert_close(portable(item.indices,item.offsets,item.per_sample_weights),output)
  chunks=[]
  for r in range(3):
   micro,_=collate_bags(item,[r],num_embeddings=5,embedding_dim=2,mode='sum' if p['weighted'] else 'mean')
   chunks.append(portable(micro.indices,micro.offsets,micro.per_sample_weights))
  torch.testing.assert_close(torch.cat(chunks),output)
print(json.dumps({'verified':True,'devices':devices,'kind':p['kind'],'packed':p['packed'],'weighted':p['weighted']}))
"""#
}
