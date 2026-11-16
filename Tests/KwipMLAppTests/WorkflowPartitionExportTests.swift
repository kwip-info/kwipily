import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

final class WorkflowPartitionExportTests:XCTestCase {
    @MainActor func testGroupPartitionPreparationAndPublishedModelReplay()async throws {
        try await verifyPortableExample(temporal:false)
    }
    @MainActor func testTemporalPartitionPreparationAndPublishedModelReplay()async throws {
        try await verifyPortableExample(temporal:true)
    }
    @MainActor private func verifyPortableExample(temporal:Bool)async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Requires actual worker")}
        // A canonical fixture path is required by indexed View's no-symlink storage.
        let root=URL(fileURLWithPath:development).appendingPathComponent(".local/portable-fixtures/"+UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let storage=root.appendingPathComponent("workspace")
        let store=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:storage,detectCapabilities:false)
        await verifyRuntimeForIntegration(store)
        XCTAssertTrue(temporal ? store.loadTemporalSplitWorkflowExample(device:"cpu") : store.loadGroupSplitWorkflowExample(device:"cpu"),store.statusMessage)
        let host=try XCTUnwrap(store.workflowSession)
        if temporal {
            let source=try XCTUnwrap(host.bindings.path(for:WorkflowExamples.sourceAddress))
            let rows=try String(contentsOfFile:source,encoding:.utf8).split(whereSeparator:\.isNewline).dropFirst()
            let pairs=try rows.map {row -> String in
                let columns=row.split(separator:",")
                let signal=try XCTUnwrap(Float(columns[2])),context=try XCTUnwrap(Float(columns[3]))
                return "\(signal.bitPattern):\(context.bitPattern)"
            }
            XCTAssertEqual(Set(pairs).count,96,"Independent temporal observations must not repeat feature pairs across partitions")
        }
        let done=expectation(description:"Actual example training")
        let observation=host.$outcomes.dropFirst().filter{!$0.isEmpty}.prefix(1).sink{_ in done.fulfill()}
        XCTAssertTrue(host.run(),host.statusMessage)
        await fulfillment(of:[done],timeout:60);observation.cancel()
        let outcome=try XCTUnwrap(host.outcomes.first)
        XCTAssertEqual(outcome.status,.completed,outcome.error ?? host.statusMessage)
        let reference=WorkflowHistoryReference(projectID:host.owner.projectID,directoryID:try XCTUnwrap(UUID(uuidString:outcome.directory.lastPathComponent)))
        guard case .verified(let archive)=WorkflowHistoryReader.load(reference,root:storage) else {return XCTFail("Example archive did not verify")}
        let expectedSources:Set<WorkflowNodeAddress>=[WorkflowExamples.sourceAddress]
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
        import csv,hashlib,json,math,os,subprocess,sys
        from datetime import datetime,timedelta
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
        def artifact(node,port):
            entry=next(a for a in after['artifacts'] if a.get('nodeID')==node and a.get('port')==port)
            return prepared_table_document(after_files,entry)
        def identities(evidence):
            return evidence['observationIDs'] if 'observationIDs' in evidence else [evidence['observationIDPrefix']+s for s in evidence['observationIDSuffixes']]
        roles=('training','validation','test')
        split={entry['port']:prepared_table_document(after_files,entry) for entry in after['artifacts'] if entry.get('nodeID')=='split'}
        assert set(split)=={*roles,'exclusions'}
        sources=json.loads((root/'source-references.json').read_text());assert len(sources)==1
        source=sources[0];source_path=root/(source['address']['localID']+'.csv')
        with source_path.open(newline='') as handle: original=list(csv.DictReader(handle))
        source_ids=[source['sha256']+':'+str(i) for i in range(len(original))]
        typed=artifact('cast','table');schema=typed['columns']
        def convert(value,dtype):
            return int(value) if dtype=='int64' else float(value) if dtype=='float64' else value
        source_rows=[[convert(row[c['name']],c['dtype']) for c in schema] for row in original]
        assert typed['rows']==source_rows and identities(typed['lineage'])==source_ids
        graph=json.loads((prep/'dataflow.json').read_text())
        node=next(n for n in graph['nodes'] if n['id']=='split')
        assert node['operator']=='kwip.data.partition'
        config=node['config'];expected_indices={role:[] for role in (*roles,'exclusions')}
        names=[c['name'] for c in schema]
        if config['strategy']=='entity':
            assert config['groupBy']==['subject_id'] and config['seed']==42
            assert config['validationPercent']==25 and config['testPercent']==25
            key_index=names.index('subject_id');groups={}
            for i,row in enumerate(source_rows):groups.setdefault(row[key_index],[]).append(i)
            def rank(key):
                typed_key=[['subject_id',schema[key_index]['dtype'],key]]
                encoded=json.dumps(['kwip.groupSplit.rank.v1',42,typed_key],sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
                return hashlib.sha256(encoded).hexdigest(),json.dumps(typed_key,separators=(',',':'),ensure_ascii=True).encode()
            ordered=sorted(groups,key=rank);percentages=(50,25,25)
            counts=[len(groups)*p//100 for p in percentages]
            for i in sorted(range(3),key=lambda i:(-(len(groups)*percentages[i]%100),i))[:len(groups)-sum(counts)]:counts[i]+=1
            cursor=0
            for role,count in zip(roles,counts):
                expected_indices[role]=sorted(i for key in ordered[cursor:cursor+count] for i in groups[key]);cursor+=count
        else:
            assert config['strategy']=='time' and config['timeColumn']=='observed_at'
            assert config['timeEncoding']=='iso8601Offset' and config['entityPolicy']=='requireDisjoint'
            assert config['entityColumns']==['subject_id'] and config['embargoMilliseconds']==0
            parse=lambda value:datetime.fromisoformat(value.replace('Z','+00:00'))
            first=parse(config['trainingEnd']);second=parse(config['validationEnd']);gap=timedelta(milliseconds=config['embargoMilliseconds'])
            assert first==parse('2026-01-02T23:00:00Z') and second==parse('2026-01-03T23:00:00Z')
            instants=[parse(row['observed_at']) for row in original]
            for i,instant in enumerate(instants):
                role='training' if instant<=first else 'validation' if first+gap<instant<=second else 'test' if instant>second+gap else 'exclusions'
                expected_indices[role].append(i)
            for role in roles:expected_indices[role].sort(key=lambda i:(instants[i],source_ids[i]))
        membership={role:set(identities(value['lineage'])) for role,value in split.items()}
        assert all(membership[a].isdisjoint(membership[b]) for i,a in enumerate(split) for b in list(split)[i+1:])
        assert set.union(*membership.values())==set(source_ids)
        for role,indices in expected_indices.items():
            value=split[role]
            assert identities(value['lineage'])==[source_ids[i] for i in indices]
            assert value['rows']==[source_rows[i] for i in indices] and value['columns']==schema
            assert value['lineage']['role']==('unassigned' if role=='exclusions' else role)
        assert not membership['exclusions'] and all(membership[role] for role in roles)
        assert len({split[role]['lineage']['partitionDigest'] for role in split})==1
        diagnostic=next(d for d in split['training']['lineage']['diagnostics'] if d.get('kind')=='partition')
        assert diagnostic['inputRows']==len(original) and diagnostic['excludedRows']==len(expected_indices['exclusions'])
        subjects={role:{original[i]['subject_id'] for i in expected_indices[role]} for role in roles}
        assert all(subjects[a].isdisjoint(subjects[b]) for a,b in [('training','validation'),('training','test'),('validation','test')])
        # Protected Test is exhaustively accounted for; inference below uses only validation.
        fit_ports={a['port'] for a in after['artifacts'] if a.get('nodeID')=='fit'}
        assert fit_ports=={*roles,'state'}
        state=artifact('fit','state');assert state['operator']=='kwip.data.preprocess'
        assert identities(state['fitEvidence'])==[source_ids[i] for i in expected_indices['training']]
        assert state['fitEvidence']['role']=='training'
        statistics={}
        for name in ('signal','context'):
            index=names.index(name);values=[source_rows[i][index] for i in expected_indices['training']]
            mean=math.fsum(values)/len(values);scale=math.sqrt(math.fsum((v-mean)**2 for v in values)/len(values)) or 1.0
            rule=next(r for r in state['rules'] if r['column']==name)
            assert rule['missing']=='error' and rule['scaling']=='standard'
            assert math.isclose(rule['mean'],mean,rel_tol=1e-12,abs_tol=1e-12)
            assert math.isclose(rule['scale'],scale,rel_tol=1e-12,abs_tol=1e-12)
            statistics[name]=(mean,scale)
        for role in roles:
            prepared=artifact('fit',role);assert identities(prepared['lineage'])==identities(split[role]['lineage'])
            assert prepared['lineage']['role']==role and len(prepared['rows'])==len(expected_indices[role])
            for before_row,prepared_row in zip(split[role]['rows'],prepared['rows']):
                for index,column in enumerate(schema):
                    if column['name'] in statistics:
                        mean,scale=statistics[column['name']]
                        assert math.isclose(prepared_row[index],(before_row[index]-mean)/scale,rel_tol=1e-12,abs_tol=1e-12)
                    else:assert prepared_row[index]==before_row[index]
        table=artifact('fit','validation')
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
