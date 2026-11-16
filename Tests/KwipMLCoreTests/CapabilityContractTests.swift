import XCTest
@testable import KwipMLCore

final class CapabilityContractTests:XCTestCase {
    let digest=String(repeating:"a",count:64)
    func tensor(_ direction:PortDirection,_ dimensions:[DimensionContract]=[.symbol("batch"),.fixed(8)],backend:ContractBackend = .pytorch)->PortContract {
        PortContract(direction == .input ? "in" : "out",direction:direction,schema:.tensor(TensorContract(dimensions,backend:backend)),phases:[.forward],maximumConnections:direction == .input ? 1 : nil)
    }
    func capability(_ ports:[PortContract],backend:ContractBackend = .pytorch)->CapabilityDescriptor {
        CapabilityDescriptor(id:"test.component",packageID:"test.package",backend:backend,ports:ports,lifecycle:LifecycleContract(phases:[.initialize,.forward]),devices:[.cpu])
    }
    func pair(_ source:CapabilityDescriptor,_ target:CapabilityDescriptor)->ContractValidation {
        CapabilityContractValidator.connection(from:ContractEndpoint(source,instanceID:"source",portID:"out"),to:ContractEndpoint(target,instanceID:"target",portID:"in"),phase:.forward)
    }
    func testVersionedPackageRoundTripAndMalformedPortFixtures() throws {
        let valid=capability([tensor(.input),tensor(.output)])
        let package=CapabilityPackageDescriptor(id:"test.package",backendVersions:["pytorch":"2.10.0"],capabilities:[valid])
        XCTAssertTrue(CapabilityContractValidator.validate(package).isValid)
        XCTAssertEqual(try JSONDecoder().decode(CapabilityPackageDescriptor.self,from:JSONEncoder().encode(package)),package)
        var malformed=valid
        malformed.ports.append(malformed.ports[0])
        XCTAssertFalse(CapabilityContractValidator.validate(malformed).isValid)
        malformed=valid;malformed.ports[0].schema.table=TableContract([])
        XCTAssertFalse(CapabilityContractValidator.validate(malformed).isValid)
        malformed=valid;malformed.ports[0].schema.tensor?.dimensions=[.fixed(-1)]
        XCTAssertFalse(CapabilityContractValidator.validate(malformed).isValid)
        var future=package;future.schemaVersion=2
        XCTAssertFalse(CapabilityContractValidator.validate(future).isValid)
        future=package;future.bundled=false
        XCTAssertFalse(CapabilityContractValidator.validate(future).isValid)
    }
    func testTensorShapeDtypeMultiplicityAndDirectionFixtures() {
        let source=capability([tensor(.output,[.fixed(4),.fixed(8)])])
        let compatible=capability([tensor(.input,[.fixed(4),.fixed(8)])])
        XCTAssertTrue(pair(source,compatible).isFullyResolved)
        XCTAssertFalse(pair(source,capability([tensor(.input,[.fixed(4),.fixed(9)])])).isValid)
        XCTAssertFalse(pair(source,capability([tensor(.input,[.symbol("square"),.symbol("square")])])).isValid)
        var wrong=compatible;wrong.ports[0].schema.tensor?.dtype = .int64
        XCTAssertFalse(pair(source,wrong).isValid)
        XCTAssertFalse(CapabilityContractValidator.connection(from:ContractEndpoint(source,instanceID:"a",portID:"out"),to:ContractEndpoint(compatible,instanceID:"b",portID:"in"),phase:.forward,existingInputConnections:1).isValid)
        XCTAssertFalse(CapabilityContractValidator.connection(from:ContractEndpoint(compatible,instanceID:"a",portID:"in"),to:ContractEndpoint(compatible,instanceID:"b",portID:"in"),phase:.forward).isValid)
        XCTAssertFalse(CapabilityContractValidator.connection(from:ContractEndpoint(source,instanceID:"a",portID:"missing"),to:ContractEndpoint(compatible,instanceID:"b",portID:"in"),phase:.forward).isValid)
    }
    func testUnknownDimensionsAreDeferredNotProvenEqual() {
        let result=pair(capability([tensor(.output)]),capability([tensor(.input)]))
        XCTAssertTrue(result.isValid);XCTAssertFalse(result.isFullyResolved)
        XCTAssertTrue(result.issues.contains {$0.severity == .deferred})
    }
    func testCrossBackendNeedsExplicitDetachedBoundary() {
        let source=capability([tensor(.output)],backend:.pytorch)
        let target=capability([tensor(.input,backend:.tensorflow)],backend:.tensorflow)
        XCTAssertFalse(pair(source,target).isValid)
        var bridge=capability([tensor(.input,backend:.pytorch),tensor(.output,backend:.tensorflow)],backend:.tensorflow)
        XCTAssertFalse(CapabilityContractValidator.validate(bridge).isValid)
        bridge.boundary=BackendBoundary(source:.pytorch,destination:.tensorflow,adapterID:"bridge.host-copy.v1")
        XCTAssertTrue(CapabilityContractValidator.validate(bridge).isValid)
        XCTAssertTrue(pair(source,bridge).isValid)
        XCTAssertTrue(pair(bridge,target).isValid)
        bridge.boundary?.detached=false
        XCTAssertFalse(CapabilityContractValidator.validate(bridge).isValid)
        bridge.boundary?.detached=true;bridge.ports[1].schema.tensor?.gradients=true
        XCTAssertFalse(CapabilityContractValidator.validate(bridge).isValid)
    }
    func testTableSchemaAndEstimatorLifecycleAreNotTensorWires() {
        let columns=[ColumnContract("value",dtype:.float64),ColumnContract("label",dtype:.int64)]
        let output=PortContract("out",direction:.output,schema:.table(TableContract(columns)),phases:[.forward],maximumConnections:nil)
        var input=PortContract("in",direction:.input,schema:.table(TableContract([columns[0]],allowsAdditionalColumns:true)),phases:[.forward])
        XCTAssertTrue(pair(capability([output],backend:.portable),capability([input],backend:.scikitLearn)).isValid)
        input.schema.table?.columns[0].nullable=false
        var nullable=output;nullable.schema.table?.columns[0].nullable=true
        XCTAssertFalse(pair(capability([nullable]),capability([input])).isValid)
        XCTAssertFalse(pair(capability([output]),capability([tensor(.input)])).isValid)
        let unfitted=PortContract("out",direction:.output,schema:.estimator(EstimatorContract(backend:.scikitLearn,family:"classifier",requiresFitted:false)),phases:[.forward],maximumConnections:nil)
        let fitted=PortContract("in",direction:.input,schema:.estimator(EstimatorContract(backend:.scikitLearn,family:"classifier")),phases:[.forward])
        XCTAssertFalse(pair(capability([unfitted],backend:.scikitLearn),capability([fitted],backend:.scikitLearn)).isValid)
    }
    func testStateOwnerAndPhaseMisuseRejected() {
        let state=StatePortContract(key:"hidden",kind:.recurrent)
        var block=capability([PortContract("in",direction:.input,schema:.state(state),phases:[.forward]),PortContract("out",direction:.output,schema:.state(state),phases:[.forward],maximumConnections:nil)])
        XCTAssertFalse(CapabilityContractValidator.validate(block).isValid)
        block.lifecycle.state=[StateDeclaration("hidden",kind:.recurrent)]
        XCTAssertTrue(CapabilityContractValidator.validate(block).isValid)
        func connection(_ target:String,_ owner:String)->ContractValidation {
            CapabilityContractValidator.connection(from:ContractEndpoint(block,instanceID:"a",portID:"out",stateOwnerID:"a"),to:ContractEndpoint(block,instanceID:target,portID:"in",stateOwnerID:owner),phase:.forward)
        }
        XCTAssertTrue(connection("a","a").isValid)
        XCTAssertFalse(connection("b","a").isValid)
        XCTAssertFalse(connection("a","other").isValid)
        XCTAssertFalse(CapabilityContractValidator.connection(from:ContractEndpoint(block,instanceID:"a",portID:"out"),to:ContractEndpoint(block,instanceID:"a",portID:"in"),phase:.fit).isValid)
    }
    func testTrainOnlyFitRequiresMembershipEvidenceAndNoHeldOutRows() {
        var transformer=capability([],backend:.scikitLearn)
        transformer.lifecycle=LifecycleContract(phases:[.fit,.transform],fitBoundary:.trainingPartitionOnly,state:[StateDeclaration("scale",kind:.fittedTransform,resetPhase:.fit)])
        var partition=PartitionEvidence(role:.training,sourceDigest:digest,membershipDigest:PartitionEvidence.membershipDigest(for:["r1","r2"]),observationIDs:["r1","r2"])
        XCTAssertTrue(CapabilityContractValidator.fit(transformer,phase:.fit,partition:partition,heldOutObservationIDs:["r3"]).isValid)
        XCTAssertFalse(CapabilityContractValidator.fit(transformer,phase:.fit,partition:partition,heldOutObservationIDs:["r2"]).isValid)
        partition.role = .validation
        XCTAssertFalse(CapabilityContractValidator.fit(transformer,phase:.fit,partition:partition,heldOutObservationIDs:[]).isValid)
        partition.role = .training;partition.membershipDigest="unknown"
        XCTAssertFalse(CapabilityContractValidator.fit(transformer,phase:.fit,partition:partition,heldOutObservationIDs:[]).isValid)
        transformer.lifecycle.fitBoundary = .notApplicable
        XCTAssertFalse(CapabilityContractValidator.validate(transformer).isValid)
    }
    func testWeightsCannotClaimFullTrainingResume() {
        var artifact=ArtifactIdentity(id:"weights-1",contract:ArtifactContract(kind:.weights,backend:.pytorch,format:"pytorch.state_dict",resume:.weightsOnly),contentDigest:digest,producerCapabilityID:"test.component",producerVersion:ContractVersion(),graphDigest:digest,includedState:[.parameters])
        XCTAssertTrue(CapabilityContractValidator.artifact(artifact).isValid)
        XCTAssertFalse(CapabilityContractValidator.artifact(artifact,requiredResume:.fullTraining).isValid)
        artifact.contract=ArtifactContract(kind:.trainingCheckpoint,backend:.pytorch,format:"kwip.training.v1",resume:.fullTraining)
        XCTAssertFalse(CapabilityContractValidator.artifact(artifact,requiredResume:.fullTraining).isValid)
        artifact.includedState=[.parameters,.optimizer,.randomGenerator,.dataCursor]
        XCTAssertTrue(CapabilityContractValidator.artifact(artifact,requiredResume:.fullTraining).isValid)
        artifact.contentDigest="../../not-a-digest"
        XCTAssertFalse(CapabilityContractValidator.artifact(artifact).isValid)
    }
}

extension CapabilityContractTests {
    func testExactRuntimePinAndMembershipDigestCannotBeInvented() {
        let descriptor=capability([tensor(.output)])
        let latest=CapabilityPackageDescriptor(id:"test.package",backendVersions:["pytorch":"latest"],capabilities:[descriptor])
        XCTAssertFalse(CapabilityContractValidator.validate(latest).isValid)
        var transformer=capability([],backend:.scikitLearn)
        transformer.lifecycle=LifecycleContract(phases:[.fit],fitBoundary:.trainingPartitionOnly)
        let evidence=PartitionEvidence(role:.training,sourceDigest:digest,membershipDigest:String(repeating:"b",count:64),observationIDs:["row1","row2"])
        XCTAssertFalse(CapabilityContractValidator.fit(transformer,phase:.fit,partition:evidence,heldOutObservationIDs:[]).isValid)
    }
    func testSymbolEquivalencePropagatesAcrossAxes() {
        let source=capability([tensor(.output,[.symbol("same"),.symbol("same"),.fixed(5)])])
        let target=capability([tensor(.input,[.symbol("other"),.fixed(4),.symbol("other")])])
        XCTAssertFalse(pair(source,target).isValid)
    }
}


extension CapabilityContractTests {
    func testMembershipDigestHasCrossLanguageGoldenAndOrderIndependence() {
        XCTAssertEqual(PartitionEvidence.membershipDigest(for:["r1","r2"]),"7cff791e8558ce2046742a25325c8281a2815235e5712b1e457b2eb2eee5e0b8")
        XCTAssertEqual(PartitionEvidence.membershipDigest(for:["r2","r1"]),PartitionEvidence.membershipDigest(for:["r1","r2"]))
        XCTAssertNotEqual(PartitionEvidence.membershipDigest(for:["ab","c"]),PartitionEvidence.membershipDigest(for:["a","bc"]))
    }
}
