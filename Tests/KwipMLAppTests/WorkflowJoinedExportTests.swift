import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

final class WorkflowJoinedExportTests:XCTestCase {
    @MainActor func testJoinedPreparationAndPublishedModelReplayWithoutOriginalWorkspace()async throws {
        try await verifyPortableExample(aggregated:false)
    }
    @MainActor func testAggregatedPreparationAndPublishedModelReplayWithoutOriginalWorkspace()async throws {
        try await verifyPortableExample(aggregated:true)
    }
    @MainActor private func verifyPortableExample(aggregated:Bool)async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Requires actual worker")}
        // A canonical fixture path is required by indexed View's no-symlink storage.
        let root=URL(fileURLWithPath:development).appendingPathComponent(".local/portable-fixtures/"+UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let storage=root.appendingPathComponent("workspace")
        let store=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:storage,detectCapabilities:false)
        await verifyRuntimeForIntegration(store)
        XCTAssertTrue(aggregated ? store.loadAggregatedWorkflowExample(device:"cpu") : store.loadJoinedWorkflowExample(device:"cpu"),store.statusMessage)
        let host=try XCTUnwrap(store.workflowSession)
        let done=expectation(description:"Actual example training")
        let observation=host.$outcomes.dropFirst().filter{!$0.isEmpty}.prefix(1).sink{_ in done.fulfill()}
        XCTAssertTrue(host.run(),host.statusMessage)
        await fulfillment(of:[done],timeout:60);observation.cancel()
        let outcome=try XCTUnwrap(host.outcomes.first)
        XCTAssertEqual(outcome.status,.completed,outcome.error ?? host.statusMessage)
        let reference=WorkflowHistoryReference(projectID:host.owner.projectID,directoryID:try XCTUnwrap(UUID(uuidString:outcome.directory.lastPathComponent)))
        guard case .verified(let archive)=WorkflowHistoryReader.load(reference,root:storage) else {return XCTFail("Example archive did not verify")}
        let expectedSources:Set<WorkflowNodeAddress>=aggregated ? [WorkflowExamples.sourceAddress] : [WorkflowExamples.sourceAddress,WorkflowExamples.joinContextAddress]
        XCTAssertEqual(Set(archive.capture.sources.map(\.address)),expectedSources)
        let package=try WorkflowHistoryReader.stageTrainedExport(reference,root:storage)
        defer {try? FileManager.default.removeItem(at:package.directory)}
        let portable=root.appendingPathComponent("portable-model")
        try package.publish(to:portable)
        let originalCheckpoint=try Data(contentsOf:outcome.directory.appendingPathComponent("training/artifacts/checkpoint.pt"))
        XCTAssertEqual(try Data(contentsOf:portable.appendingPathComponent("checkpoint.pt")),originalCheckpoint)
        XCTAssertEqual(try Data(contentsOf:portable.appendingPathComponent("model.py")),try Data(contentsOf:outcome.directory.appendingPathComponent("training/artifacts/model.py")))
        try FileManager.default.copyItem(at:outcome.directory.appendingPathComponent("training/artifacts/predictions.json"),to:root.appendingPathComponent("expected-predictions.json"))
        let prep=root.appendingPathComponent("portable-preparation")
        try FileManager.default.copyItem(at:outcome.directory.appendingPathComponent("preparation/artifacts"),to:prep)
        var bindings:[String:String]=[:]
        for source in archive.capture.sources {
            let target=root.appendingPathComponent(source.address.localID+".csv")
            let bytes=try Data(contentsOf:outcome.directory.appendingPathComponent(source.path))
            XCTAssertEqual(WorkflowArchiveCodec.hash(bytes),source.sha256)
            try bytes.write(to:target)
            bindings[source.address.localID]=target.path
        }
        try JSONEncoder().encode(bindings).write(to:root.appendingPathComponent("bindings.json"))
        // Keep an explicit portable source manifest; model export itself excludes raw sources.
        try JSONEncoder().encode(archive.capture.sources).write(to:root.appendingPathComponent("source-references.json"))
        let python=try XCTUnwrap(store.runtime).executableURL
        store.newProject()
        try FileManager.default.removeItem(at:storage)
        let script=root.appendingPathComponent("verify.py")
        try #"""
        import hashlib,json,os,subprocess,sys
        from pathlib import Path
        import torch
        root=Path(__file__).parent
        assert not (root/'workspace'/'workflow-runs').exists()
        assert not (root/'workspace'/'example-sources').exists()
        assert all(not (Path(p)/'runner'/'dataflow.py').exists() for p in sys.path if p)
        for source in json.loads((root/'source-references.json').read_text()):
            path=root/(source['address']['localID']+'.csv')
            assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
        prep=root/'portable-preparation';replay=root/'replayed'
        subprocess.run([sys.executable,str(prep/'prep.py'),'--source-bindings',str(root/'bindings.json'),'--output',str(replay)],check=True,stdout=subprocess.DEVNULL)
        sys.path.insert(0,str(prep))
        from dataflow_storage import completed_inventory,prepared_table_document
        before,before_files,_=completed_inventory(prep/'manifest.json',lazy=True)
        after,after_files,_=completed_inventory(replay/'manifest.json',lazy=True)
        def scientific_outputs(manifest,files):
            result={}
            for entry in manifest['artifacts']:
                if entry['kind']=='table':
                    value=prepared_table_document(files,entry)
                    lineage=value['lineage']
                    if 'observationIDPrefix' in lineage:
                        lineage=dict(lineage)
                        prefix=lineage.pop('observationIDPrefix');suffixes=lineage.pop('observationIDSuffixes')
                        lineage['observationIDs']=[prefix+suffix for suffix in suffixes]
                        value={**value,'lineage':lineage}
                    result[(entry['nodeID'],entry['port'],'table')]=value
                elif entry['kind']=='state':result[(entry['nodeID'],entry['port'],'state')]=json.loads(files[entry['path']])
            return result
        assert scientific_outputs(before,before_files)==scientific_outputs(after,after_files)
        def programs(manifest):
            return {entry['path']:entry for entry in manifest['artifacts'] if entry['kind'] not in ('table','state','tableSegment','tableViewIndex')}
        before_programs,after_programs=programs(before),programs(after)
        assert before_programs==after_programs,[(name,before_programs.get(name,{}).get("kind")) for name in sorted(set(before_programs)|set(after_programs)) if before_programs.get(name)!=after_programs.get(name)]
        from table_view import query_index
        def verified_indexes(directory,manifest,files):
            pages={}
            for entry in manifest['artifacts']:
                if entry['kind']!='tableViewIndex':continue
                table=next(item for item in manifest['artifacts'] if item['kind']=='table' and (item['nodeID'],item['port'])==(entry['nodeID'],entry['port']))
                expected=prepared_table_document(files,table)
                assert len(expected['rows'])<=100,'Fixture needs complete bounded-page verification'
                assert entry['sourceSHA256']==table['sha256']
                page=query_index(directory/entry['path'],entry['sha256'],source_sha256=table['sha256'],limit=100)
                assert page['sourceSHA256']==table['sha256'] and page['cellTruncated'] is False
                assert page['rows']==expected['rows'] and page['columns']==expected['columns']
                assert page['positions']==list(range(len(expected['rows'])))
                assert page['totalRows']==page['matchedRows']==len(expected['rows']) and page['nextOffset'] is None
                pages[(entry['nodeID'],entry['port'])]={key:page[key] for key in ('columns','rows','positions','totalRows')}
            return pages
        assert verified_indexes(prep,before,before_files)==verified_indexes(replay,after,after_files)
        aggregated=next((a for a in after['artifacts'] if a.get('nodeID')=='aggregate' and a.get('port')=='table'),None)
        if aggregated:
            summaries=prepared_table_document(after_files,aggregated)
            names=[c['name'] for c in summaries['columns']]
            assert len(summaries['rows'])==64
            assert len({r[names.index('subject_id')] for r in summaries['rows']})==64
            assert all(r[names.index('label')]==r[names.index('label_check')] and r[names.index('measurement_count')]==3 for r in summaries['rows'])
        validation=next(a for a in after['artifacts'] if a.get('nodeID')=='fit' and a.get('port')=='validation')
        table=prepared_table_document(after_files,validation)
        columns=[c['name'] for c in table['columns']]
        inputs=[[row[columns.index(name)] for name in ['signal','context']] for row in table['rows']]
        expected=json.loads((root/'expected-predictions.json').read_text())['predictionRows']
        values=torch.tensor(inputs,dtype=torch.float32)
        torch.testing.assert_close(values,torch.tensor([r['input'] for r in expected]),rtol=0,atol=0)
        portable=root/'portable-model';sys.path.insert(0,str(portable))
        from inference import load_model
        from model import Model
        checkpoint=torch.load(portable/'checkpoint.pt',weights_only=True,map_location='cpu')
        reference=Model();reference.load_state_dict(checkpoint['modelState'],strict=True);reference.eval()
        predictor=load_model()
        for size in [1,7,16]:
            actual=predictor.predict(values,batch_size=size)
            with torch.inference_mode(): wanted=reference(values)
            torch.testing.assert_close(actual,wanted,rtol=1e-5,atol=1e-6)
            assert actual.argmax(1).tolist()==[row['predicted'] for row in expected]
        print(json.dumps({'verified':True,'rows':len(inputs),'sources':len(json.loads((root/'source-references.json').read_text())),'batchSizes':[1,7,16]}))
        """#.write(to:script,atomically:true,encoding:.utf8)
        let verified=expectation(description:"Independent preparation and exported inference")
        var code:Int32?,lines:[String]=[]
        let process=try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:"/usr/bin/env"),
            ["-u","PYTHONPATH","-u","PYTHONHOME","PYTHONNOUSERSITE=1",python.path,script.path],root,
            {lines.append($0)},{value,error in code=value;if !error.isEmpty {lines.append(error)};verified.fulfill()})
        defer {process.cancel()}
        await fulfillment(of:[verified],timeout:45)
        XCTAssertEqual(code,0,lines.joined(separator:"\n"))
        XCTAssertTrue(lines.contains {$0.contains("\"verified\": true")},lines.joined(separator:"\n"))
    }
}
