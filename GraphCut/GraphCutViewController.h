//
//  GraphCutViewController.h
//  GraphCut
//
//  Created by Agent Link on 14-7-7.
//  Copyright (c) 2014年 link. All rights reserved.
//

#import <UIKit/UIKit.h>

@interface GraphCutViewController : UIViewController<UIImagePickerControllerDelegate, UINavigationControllerDelegate,UIPopoverControllerDelegate>

@property (strong,nonatomic) UIImagePickerController *imgPickerControll;
@property (weak, nonatomic) IBOutlet UIImageView *imageView;

- (IBAction)openPicture:(id)sender;
- (IBAction)switchForeground:(id)sender;
- (IBAction)switchBackground:(id)sender;
- (IBAction)done:(id)sender;


@end
