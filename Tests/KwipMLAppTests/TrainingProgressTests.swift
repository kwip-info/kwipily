import XCTest
@testable import KwipMLApp

final class TrainingProgressTests:XCTestCase {
    let worker="82E8D1F2-2CEE-44D3-81CB-22D7A157C005"
    func point(_ seq:Int=1,batch:Int=1,update:Int=1,kind:String="training",epoch:Int=1)throws->TrainingProgressPoint {
        var fields:[String:Any] = ["event":"trainingProgress","schemaVersion":1,"runID":worker,"sequence":seq,"epoch":epoch,"epochBatch":batch,"epochBatches":10000,"batches":batch,"updates":update,"elapsedSeconds":Double(batch),"kind":kind,"trainingLoss":2.0]
        if kind=="training" {fields["numerator"]=4.0;fields["denominator"]=2.0}
        else {fields["validationLoss"]=1.5;fields["accuracy"]=0.75}
        return try TrainingProgressPoint.decode(JSONSerialization.data(withJSONObject:fields))
    }
    func testOwnedMonotonicProgressAndNoFabricatedValidation()throws {
        var ledger=TrainingProgressLedger()
        let first=try point(update:0)
        try ledger.append(first,worker:worker,maximumEpoch:2)
        XCTAssertThrowsError(try ledger.append(first,worker:worker,maximumEpoch:2))
        XCTAssertThrowsError(try ledger.append(point(2,batch:2),worker:UUID().uuidString,maximumEpoch:2))
        XCTAssertThrowsError(try ledger.append(point(2,batch:2,kind:"validation"),worker:worker,maximumEpoch:2))
        try ledger.append(point(2,batch:2),worker:worker,maximumEpoch:2)
        XCTAssertEqual(ledger.points.count,2)
        XCTAssertTrue(TrainingChartModel.values(metrics:[],progress:ledger.points,axis:.updates,metric:.accuracy,smoothed:false).isEmpty)
    }
    func testLossDenominatorNullAndUnknownFields()throws {
        let row=try point();var wire=try JSONSerialization.jsonObject(with:JSONEncoder().encode(row)) as! [String:Any]
        wire["trainingLoss"]=999
        XCTAssertThrowsError(try TrainingProgressPoint.decode(JSONSerialization.data(withJSONObject:wire)))
        wire["trainingLoss"]=NSNull();wire["denominator"]=0;wire["numerator"]=0
        XCTAssertNil(try TrainingProgressPoint.decode(JSONSerialization.data(withJSONObject:wire)).trainingLoss)
        wire["unknown"]=true
        XCTAssertThrowsError(try TrainingProgressPoint.decode(JSONSerialization.data(withJSONObject:wire)))
    }
    func testLongProgressRemainsBoundedAndTerminalSurvives()throws {
        var ledger=TrainingProgressLedger()
        for i in 1...10000 {try ledger.append(point(i,batch:i,update:i/4),worker:worker,maximumEpoch:1)}
        try ledger.append(point(10001,batch:10000,update:2500,kind:"validation"),worker:worker,maximumEpoch:1)
        XCTAssertLessThanOrEqual(ledger.points.count,TrainingProgressLedger.maximumPoints)
        XCTAssertEqual(ledger.points.first?.sequence,1);XCTAssertEqual(ledger.points.last?.sequence,10001)
        let archive=TrainingProgressArchive(schemaVersion:1,requestIdentity:"request",workerRunID:worker,policy:.init(),observed:ledger.observed,points:ledger.points)
        let decoded=try JSONDecoder().decode(TrainingProgressArchive.self,from:JSONEncoder().encode(archive))
        try decoded.validate(request:"request",worker:worker,metrics:[.init(epoch:1,trainLoss:2,validationLoss:1.5,accuracy:0.75)])
        XCTAssertThrowsError(try decoded.validate(request:"other",worker:worker,metrics:[]))
    }
    func testOldEpochArchivesAndRenderOnlySmoothing()throws {
        let metrics:[EpochMetric]=[.init(epoch:1,trainLoss:4,validationLoss:3,accuracy:0.5),.init(epoch:2,trainLoss:2,validationLoss:1,accuracy:0.75)]
        let raw=TrainingChartModel.values(metrics:metrics,progress:[],axis:.epoch,metric:.loss,smoothed:false)
        let smooth=TrainingChartModel.values(metrics:metrics,progress:[],axis:.epoch,metric:.loss,smoothed:true)
        XCTAssertEqual(raw.filter {$0.series=="Validation"},smooth.filter {$0.series=="Validation"})
        XCTAssertEqual(smooth.last(where:{$0.series=="Training"})?.value,3.6)
        XCTAssertEqual(metrics.last?.trainLoss,2)
        XCTAssertTrue(TrainingChartModel.values(metrics:metrics,progress:[],axis:.updates,metric:.loss,smoothed:false).isEmpty)
        XCTAssertEqual(TrainingChartModel.domain([]),0...1)
        XCTAssertEqual(TrainingChartModel.window(0...100,lower:0.25,upper:0.75),25...75)
    }
    func testRenderingPreservesPeakAndEndpoints() {
        let values=(0..<10000).map {i in TrainingChartValue(id:String(i),x:Double(i),value:i==4567 ? 100:Double(i%3),series:"Training",epoch:1,batches:i,updates:i,elapsed:Double(i))}
        let rendered=TrainingChartModel.render(values)
        XCTAssertLessThanOrEqual(rendered.count,400)
        XCTAssertTrue(rendered.contains(values[4567]));XCTAssertTrue(rendered.contains(values[0]));XCTAssertTrue(rendered.contains(values[9999]))
    }
    func testLiveEpochAndProgressUseIdenticalNumericDecoding()throws {
        // Foundation's JSONSerialization rounds this exact emitted value
        // differently from JSONDecoder. Both evidence paths must use one reader.
        let epoch=try TrainingEpochObservation.decode(Data("{\"event\":\"epoch\",\"epoch\":1,\"trainLoss\":2.0,\"validationLoss\":0.014526826858520508,\"accuracy\":1.0}".utf8))
        var wire=try JSONSerialization.jsonObject(with:JSONEncoder().encode(point(2,batch:10000,kind:"validation"))) as! [String:Any]
        wire["accuracy"]=1.0
        let prefix=String(decoding:try JSONSerialization.data(withJSONObject:wire),as:UTF8.self)
        let bytes=Data(prefix.replacingOccurrences(of:"1.5",with:"0.014526826858520508").utf8)
        let progress=try TrainingProgressPoint.decode(bytes)
        XCTAssertEqual(epoch.validationLoss,progress.validationLoss)
        XCTAssertEqual(epoch.accuracy,progress.accuracy)
    }
    func testReportingPolicyStrictAndRoundTrips()throws {
        XCTAssertEqual(try TrainingReportingPolicy.decode(JSONEncoder().encode(TrainingReportingPolicy())),.init())
        for wire in ["{}","{\"everyBatches\":true,\"everySeconds\":1}","{\"everyBatches\":0,\"everySeconds\":1}","{\"everyBatches\":1,\"everySeconds\":0}"] {XCTAssertThrowsError(try TrainingReportingPolicy.decode(Data(wire.utf8)))}
    }
}
