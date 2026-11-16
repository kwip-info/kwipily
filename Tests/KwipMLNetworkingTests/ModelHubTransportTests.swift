import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLNetworking

private final class HubFixture:URLProtocol {
    static var status=200
    static var body=Data("{}".utf8)
    static var range:String?
    override class func canInit(with request:URLRequest)->Bool {true}
    override class func canonicalRequest(for request:URLRequest)->URLRequest {request}
    override func startLoading() {
        var fields=["Content-Type":"application/octet-stream"]
        if let range=Self.range {fields["Content-Range"]=range}
        let response=HTTPURLResponse(url:request.url!,statusCode:Self.status,httpVersion:nil,headerFields:fields)!
        client?.urlProtocol(self,didReceive:response,cacheStoragePolicy:.notAllowed)
        client?.urlProtocol(self,didLoad:Self.body);client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}
final class ModelHubTransportTests:XCTestCase {
    private func transport()throws->ModelHubTransport {
        #if DEBUG
        let result=ModelHubTransport();result.testProtocolClasses=[HubFixture.self];return result
        #else
        throw XCTSkip("Fixture injection is debug only")
        #endif
    }
    func testExplicitPublicNetworkSnapshot()async throws {
        guard ProcessInfo.processInfo.environment["KWIPML_TEST_MODEL_HUB_NETWORK"]=="1" else {throw XCTSkip("Opt-in actual public network check")}
        let data=try await ModelHubTransport().perform(.init(action:.resolve,repository:"hf-internal-testing/tiny-random-gpt2",revision:"main"))
        let object=try JSONSerialization.jsonObject(with:data) as? [String:Any]
        XCTAssertEqual((object?["sha"] as? String)?.count,40)
        let weights=try await ModelHubTransport().perform(.init(action:.file,repository:"hf-internal-testing/tiny-random-gpt2",revision:"71034c5d8bde858ff824298bdedc65515b97d2b9",file:"model.safetensors",offset:0,total:453864))
        XCTAssertEqual(weights.count,453864)
    }
    func testExactSmallFileAndTruncatedBodyRefusal()async throws {
        let transport=try transport(),request=ModelHubRequest(action:.file,repository:"org/model",revision:String(repeating:"a",count:40),file:"config.json",offset:0,total:2)
        HubFixture.status=200;HubFixture.range=nil;HubFixture.body=Data("{}".utf8)
        let data=try await transport.perform(request);XCTAssertEqual(data,HubFixture.body)
        HubFixture.body=Data("x".utf8)
        do {_=try await transport.perform(request);XCTFail("Truncated file accepted")} catch {}
        HubFixture.body=Data("xxx".utf8)
        do {_=try await transport.perform(request);XCTFail("Oversized file accepted")} catch {}
    }
    func testRangeAndAuthenticationFailures()async throws {
        let transport=try transport(),request=ModelHubRequest(action:.file,repository:"org/model",revision:String(repeating:"a",count:40),file:"config.json",offset:0,total:2)
        HubFixture.status=206;HubFixture.body=Data("{}".utf8);HubFixture.range="bytes 0-1/2"
        _=try await transport.perform(request)
        HubFixture.range="bytes 0-1/3"
        do {_=try await transport.perform(request);XCTFail("Wrong range accepted")} catch {}
        HubFixture.status=401
        do {_=try await transport.perform(request);XCTFail("Unauthenticated download accepted")} catch NetworkBrokerFailure.signInRequired {} catch {XCTFail("Wrong authentication failure")}
    }
}
