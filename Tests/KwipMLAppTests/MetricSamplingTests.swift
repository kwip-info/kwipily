import XCTest
@testable import KwipMLApp

final class MetricSamplingTests: XCTestCase {
    func testChartSamplingPreservesEndpointsAndExactOriginalValues() {
        let raw=Array(0..<10000)
        let sampled=MetricSampling.sample(raw)
        XCTAssertEqual(sampled.count,250)
        XCTAssertEqual(sampled.first,0);XCTAssertEqual(sampled.last,9999)
        XCTAssertEqual(Set(sampled).count,sampled.count)
        XCTAssertEqual(sampled,sampled.sorted())
        XCTAssertTrue(sampled.allSatisfy(raw.contains))
    }
    func testShortSeriesAreUntouchedAndDegenerateLimitsAreBounded() {
        for count in [0,1,2,249,250] { let raw=Array(0..<count);XCTAssertEqual(MetricSampling.sample(raw),raw) }
        XCTAssertEqual(MetricSampling.sample([1,2,3],limit:0),[])
        XCTAssertEqual(MetricSampling.sample([1,2,3],limit:1),[3])
        XCTAssertEqual(MetricSampling.sample([1,2,3],limit:2),[1,3])
    }
    func testPublicationScheduleBoundsWorkAndKeepsFirstLast() {
        for total in [1,5,100,200,201,250,999,10000] {
            let published=(1...total).filter {MetricSampling.shouldPublish(epoch:$0,totalEpochs:total)}
            XCTAssertLessThanOrEqual(published.count,201,"epochs=\(total)")
            XCTAssertEqual(published.first,1);XCTAssertEqual(published.last,total)
            if total <= 200 {XCTAssertEqual(published.count,total)}
        }
        XCTAssertFalse(MetricSampling.shouldPublish(epoch:0,totalEpochs:100))
        XCTAssertFalse(MetricSampling.shouldPublish(epoch:101,totalEpochs:100))
        XCTAssertFalse(MetricSampling.shouldPublish(epoch:1,totalEpochs:0))
    }
}
