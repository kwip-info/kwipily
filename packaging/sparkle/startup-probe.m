// Developer-only synchronous SDK configuration probe; never shipped in the app.
#import <AppKit/AppKit.h>
#import <Sparkle/Sparkle.h>

int main(void) {@autoreleasepool {
    [NSApplication sharedApplication];
    SPUStandardUpdaterController *controller = [[SPUStandardUpdaterController alloc]
        initWithStartingUpdater:NO updaterDelegate:nil userDriverDelegate:nil];
    NSError *error = nil;
    BOOL started = [controller.updater startUpdater:&error];
    NSDictionary *result = @{@"started":@(started), @"error":error.localizedDescription ?: @"",
        @"domain":error.domain ?: @"", @"code":@(error.code)};
    NSData *data = [NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingSortedKeys error:nil];
    fwrite(data.bytes, 1, data.length, stdout); fputc('\n', stdout);
    // startUpdater queues its automatic cycle asynchronously. Do not service the
    // run loop, check a feed, download an installer, or execute that cycle here.
    return 0;
}}
