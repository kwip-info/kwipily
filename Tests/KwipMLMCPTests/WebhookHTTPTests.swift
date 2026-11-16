import XCTest
import KwipMLCore
@testable import KwipMLMCP

final class WebhookHTTPTests:XCTestCase {
    func testWebhookParserUsesDistinctBoundedPOSTRouteAndRejectsAmbiguousFraming()throws {
        let path="/hooks/"+UUID().uuidString
        let text="POST \(path) HTTP/1.1\r\nHost: 127.0.0.1:8788\r\nContent-Length: 2\r\nContent-Type: application/json\r\n\r\n{}"
        var parser=MCPHTTPParser(webhook:true)
        XCTAssertNil(try parser.append(Data(text.utf8.prefix(20))))
        XCTAssertNotNil(try parser.append(Data(text.utf8.dropFirst(20))))
        for invalid in [text.replacingOccurrences(of:"POST",with:"GET"),text.replacingOccurrences(of:path,with:"/mcp"),text.replacingOccurrences(of:"Content-Length: 2",with:"Content-Length: 2\r\nContent-Length: 2"),text+"smuggled",text.replacingOccurrences(of:path,with:"/hooks/../file"),text.replacingOccurrences(of:"Content-Length: 2",with:"Content-Length: 1048577")] {
            var p=MCPHTTPParser(webhook:true);XCTAssertThrowsError(try p.append(Data(invalid.utf8)))
        }
        var mcp=MCPHTTPParser();XCTAssertThrowsError(try mcp.append(Data(text.utf8)))
    }
}

extension WebhookHTTPTests {
    func testActualLoopbackHTTPBrokerAndBrowserOriginRefusal()async throws {
        let port=Int.random(in:30000...50000),id=UUID().uuidString
        let broker=try WebhookLocalHTTPBroker(port:port){data in
            let request=try JSONDecoder().decode(KwipMLCore.WebhookHTTPRequest.self,from:data)
            guard request.body==Data("{}".utf8) else{throw HTTPRefusal(status:422)}
            return try JSONEncoder().encode(KwipMLCore.WebhookHTTPResponse(status:202,code:"accepted"))
        }
        defer{broker.stop()}
        let ready=expectation(description:"Loopback listener")
        broker.start{actual,error in XCTAssertEqual(actual,UInt16(port));XCTAssertNil(error);ready.fulfill()}
        await fulfillment(of:[ready],timeout:5)
        var request=URLRequest(url:URL(string:"http://127.0.0.1:\(port)/hooks/\(id)")!);request.httpMethod="POST";request.httpBody=Data("{}".utf8);request.setValue("application/json",forHTTPHeaderField:"Content-Type");request.timeoutInterval=5
        let (_,accepted)=try await URLSession.shared.data(for:request);XCTAssertEqual((accepted as? HTTPURLResponse)?.statusCode,202)
        request.setValue("https://untrusted.example",forHTTPHeaderField:"Origin")
        let (_,refused)=try await URLSession.shared.data(for:request);XCTAssertEqual((refused as? HTTPURLResponse)?.statusCode,403)
    }
}
