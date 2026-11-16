// Developer-only actual XPC download probe. Never embedded in customer releases.
#import <Foundation/Foundation.h>
#import "SPUDownloaderProtocol.h"
#import "SPUDownloaderDelegate.h"
#import "SPUDownloadData.h"

@interface DownloadProbe : NSObject <SPUDownloaderDelegate>
@property BOOL done;
@property BOOL success;
@property NSUInteger length;
@end
@implementation DownloadProbe
- (void)downloaderDidSetDownloadBookmarkData:(NSData *)data downloadToken:(NSString *)token {}
- (void)downloaderDidReceiveExpectedContentLength:(int64_t)length {}
- (void)downloaderDidReceiveDataOfLength:(uint64_t)length {}
- (void)downloaderDidFinishWithTemporaryDownloadData:(SPUDownloadData *)data {self.success=YES;self.length=data.data.length;self.done=YES;}
- (void)downloaderDidFailWithError:(NSError *)error {self.success=NO;self.done=YES;}
@end

int main(int argc,const char **argv) {@autoreleasepool {
    if(argc!=2)return 2;
    DownloadProbe *probe=[DownloadProbe new];
    NSXPCConnection *connection=[[NSXPCConnection alloc] initWithServiceName:@"info.kwip.KwipML.Sparkle.DownloaderService"];
    connection.remoteObjectInterface=[NSXPCInterface interfaceWithProtocol:@protocol(SPUDownloaderProtocol)];
    connection.exportedInterface=[NSXPCInterface interfaceWithProtocol:@protocol(SPUDownloaderDelegate)];
    connection.exportedObject=probe;
    [connection resume];
    id<SPUDownloaderProtocol> remote=[connection remoteObjectProxyWithErrorHandler:^(NSError *error){probe.done=YES;}];
    NSMutableURLRequest *request=[NSMutableURLRequest requestWithURL:[NSURL URLWithString:[NSString stringWithUTF8String:argv[1]]]];
    request.timeoutInterval=8;
    [request setValue:@"Bearer synthetic-probe-not-a-real-credential" forHTTPHeaderField:@"Authorization"];
    [remote startTemporaryDownloadWithRequest:request];
    NSDate *deadline=[NSDate dateWithTimeIntervalSinceNow:12];
    while(!probe.done && deadline.timeIntervalSinceNow>0)[[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.01]];
    printf("{\"completed\":%s,\"success\":%s,\"length\":%lu}\n",probe.done?"true":"false",probe.success?"true":"false",(unsigned long)probe.length);
    __block BOOL cleaned=NO;
    [remote cleanup:^{cleaned=YES;}];
    NSDate *cleanupDeadline=[NSDate dateWithTimeIntervalSinceNow:2];
    while(!cleaned && cleanupDeadline.timeIntervalSinceNow>0)[[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.01]];
    [connection invalidate];
    return probe.done?0:1;
}}
