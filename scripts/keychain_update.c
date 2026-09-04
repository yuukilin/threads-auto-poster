#include <CoreFoundation/CoreFoundation.h>
#include <Cocoa/Cocoa.h>
#include <Security/Security.h>

#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define TOKEN_CAPACITY 16384

static void secure_zero(void *value, size_t length) {
    volatile unsigned char *cursor = value;
    while (length-- > 0) *cursor++ = 0;
}

static CFStringRef make_string(const char *value) {
    return CFStringCreateWithCString(
        kCFAllocatorDefault, value, kCFStringEncodingUTF8
    );
}

static CFMutableDictionaryRef make_query(
    CFStringRef account, CFStringRef service
) {
    CFMutableDictionaryRef query = CFDictionaryCreateMutable(
        kCFAllocatorDefault,
        0,
        &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks
    );
    if (query != NULL) {
        CFDictionarySetValue(query, kSecClass, kSecClassGenericPassword);
        CFDictionarySetValue(query, kSecAttrAccount, account);
        CFDictionarySetValue(query, kSecAttrService, service);
    }
    return query;
}

static int write_all(const unsigned char *bytes, CFIndex length) {
    CFIndex written = 0;
    while (written < length) {
        ssize_t amount = write(
            STDOUT_FILENO, bytes + written, (size_t)(length - written)
        );
        if (amount <= 0) return 1;
        written += amount;
    }
    return 0;
}

static size_t prompt_for_token(unsigned char *buffer, size_t capacity) {
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];

        NSAlert *alert = [[NSAlert alloc] init];
        alert.messageText = @"貼上 Threads API 存取權杖";
        alert.informativeText = @"權杖只會存進這台 Mac 的鑰匙圈，不會顯示、上傳或寫入檔案。";
        [alert addButtonWithTitle:@"安全儲存"];
        [alert addButtonWithTitle:@"取消"];

        NSSecureTextField *field = [[NSSecureTextField alloc]
            initWithFrame:NSMakeRect(0, 0, 440, 26)];
        field.placeholderString = @"在這裡貼上權杖";
        alert.accessoryView = field;

        [NSApp activateIgnoringOtherApps:YES];
        NSModalResponse response = [alert runModal];
        if (response != NSAlertFirstButtonReturn) return 0;

        NSData *data = [field.stringValue dataUsingEncoding:NSUTF8StringEncoding];
        if (data.length == 0 || data.length >= capacity) return 0;
        memcpy(buffer, data.bytes, data.length);
        return data.length;
    }
}

int main(int argc, char **argv) {
    const bool is_update = argc == 5 && strcmp(argv[1], "update") == 0;
    const bool is_prompt = argc == 5 && strcmp(argv[1], "prompt-update") == 0;
    const bool is_read = argc == 4 && strcmp(argv[1], "read") == 0;
    const bool is_delete = argc == 4 && strcmp(argv[1], "delete") == 0;
    if (!is_update && !is_prompt && !is_read && !is_delete) {
        fputs(
            "usage: keychain_update update ACCOUNT SERVICE LABEL\n"
            "       keychain_update prompt-update ACCOUNT SERVICE LABEL\n"
            "       keychain_update read ACCOUNT SERVICE\n"
            "       keychain_update delete ACCOUNT SERVICE\n",
            stderr
        );
        return 2;
    }

    int exit_code = 1;
    OSStatus status = errSecParam;
    unsigned char token[TOKEN_CAPACITY] = {0};
    size_t token_length = 0;
    CFStringRef account = make_string(argv[2]);
    CFStringRef service = make_string(argv[3]);
    CFStringRef label = (is_update || is_prompt) ? make_string(argv[4]) : NULL;
    CFDataRef token_data = NULL;
    CFMutableDictionaryRef query = NULL;

    if (account == NULL || service == NULL ||
        ((is_update || is_prompt) && label == NULL)) {
        goto cleanup;
    }
    query = make_query(account, service);
    if (query == NULL) goto cleanup;

    if (is_read) {
        CFTypeRef result = NULL;
        CFDictionarySetValue(query, kSecReturnData, kCFBooleanTrue);
        CFDictionarySetValue(query, kSecMatchLimit, kSecMatchLimitOne);
        status = SecItemCopyMatching(query, &result);
        if (status == errSecSuccess && result != NULL) {
            CFDataRef result_data = (CFDataRef)result;
            exit_code = write_all(
                CFDataGetBytePtr(result_data), CFDataGetLength(result_data)
            );
            CFRelease(result);
        }
        goto cleanup;
    }

    if (is_delete) {
        status = SecItemDelete(query);
        exit_code = status == errSecSuccess ? 0 : 1;
        goto cleanup;
    }

    ssize_t received = 0;
    if (is_prompt) {
        token_length = prompt_for_token(token, sizeof(token));
    } else {
        while ((received = read(
            STDIN_FILENO,
            token + token_length,
            sizeof(token) - token_length
        )) > 0) {
            token_length += (size_t)received;
            if (token_length == sizeof(token)) break;
        }
    }
    if (received < 0 || token_length == 0 || token_length == sizeof(token)) {
        fputs("invalid token input\n", stderr);
        exit_code = 2;
        goto cleanup;
    }
    token_data = CFDataCreate(
        kCFAllocatorDefault, token, (CFIndex)token_length
    );
    if (token_data == NULL) goto cleanup;

    CFMutableDictionaryRef update = CFDictionaryCreateMutable(
        kCFAllocatorDefault,
        0,
        &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks
    );
    if (update == NULL) goto cleanup;
    CFDictionarySetValue(update, kSecValueData, token_data);
    CFDictionarySetValue(update, kSecAttrLabel, label);
    status = SecItemUpdate(query, update);
    CFRelease(update);

    if (status == errSecItemNotFound) {
        CFDictionarySetValue(query, kSecValueData, token_data);
        CFDictionarySetValue(query, kSecAttrLabel, label);
        status = SecItemAdd(query, NULL);
    }
    exit_code = status == errSecSuccess ? 0 : 1;

cleanup:
    if (exit_code != 0 && status != errSecSuccess) {
        fprintf(stderr, "keychain operation failed (%d)\n", (int)status);
    }
    if (query != NULL) CFRelease(query);
    if (token_data != NULL) CFRelease(token_data);
    if (label != NULL) CFRelease(label);
    if (service != NULL) CFRelease(service);
    if (account != NULL) CFRelease(account);
    secure_zero(token, sizeof(token));
    return exit_code;
}
