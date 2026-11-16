import Foundation
import KwipMLCore

/// Reuses the existing network-only helper; weights never enter the host's network stack.
public final class ModelHubTransport:NSObject,URLSessionTaskDelegate {
    private var ocr=false
    private func allowed(_ url:URL)->Bool {ocr ? OCRPackCatalog.allowed(url):ModelHubRequest.allowedRedirect(url)}
    #if DEBUG
    var testProtocolClasses:[AnyClass]?
    #endif
    public func urlSession(_ session:URLSession,task:URLSessionTask,willPerformHTTPRedirection response:HTTPURLResponse,newRequest request:URLRequest,completionHandler:@escaping(URLRequest?)->Void) {
        guard task.countOfBytesReceived==0,let url=request.url,allowed(url),task.currentRequest?.url != url else {completionHandler(nil);return}
        var next=request
        if url.host != "huggingface.co" {next.setValue(nil,forHTTPHeaderField:"Authorization")}
        completionHandler(next)
    }
    public func perform(_ model:ModelHubRequest)async throws->Data {
        ocr=model.action == .ocrFile
        let request=try model.prepared(),configuration=URLSessionConfiguration.ephemeral

        #if DEBUG
        configuration.protocolClasses=testProtocolClasses
        #endif
        configuration.httpCookieStorage=nil;configuration.urlCache=nil;configuration.httpShouldSetCookies=false
        configuration.timeoutIntervalForResource=45
        let session=URLSession(configuration:configuration,delegate:self,delegateQueue:nil)
        defer {session.invalidateAndCancel()}
        let (bytes,response)=try await session.bytes(for:request)
        guard let http=response as? HTTPURLResponse,let final=http.url,allowed(final) else {throw NetworkBrokerFailure.unavailable}
        if http.statusCode==401 || http.statusCode==403 {throw NetworkBrokerFailure.signInRequired}
        if model.action == .resolve {
            guard http.statusCode==200,final.host=="huggingface.co" else {throw NetworkBrokerFailure.unavailable}
        } else {
            let start=model.offset!,total=model.total!,end=start+Int64(model.maximumResponseBytes)-1
            let validRange=http.statusCode==206 && http.value(forHTTPHeaderField:"Content-Range")=="bytes \(start)-\(end)/\(total)"
            let validWhole=http.statusCode==200 && start==0 && total<=ModelHubRequest.chunkBytes
            guard validRange || validWhole else {throw NetworkBrokerFailure.unavailable}
        }
        guard response.expectedContentLength<=model.maximumResponseBytes else {throw NetworkBrokerFailure.unavailable}
        var data=Data();data.reserveCapacity(min(model.maximumResponseBytes,1_048_576))
        for try await byte in bytes {try Task.checkCancellation();guard data.count<model.maximumResponseBytes else {throw NetworkBrokerFailure.unavailable};data.append(byte)}
        if model.action != .resolve, data.count != model.maximumResponseBytes {throw NetworkBrokerFailure.unavailable}
        return data
    }
}
